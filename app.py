import os
import base64
import time
from io import BytesIO
from google import genai

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request

app = Flask(__name__)

# ====== Gemini настройка ======
GEMINI_MODEL = "gemini-2.5-flash"

try:
    gemini_client = genai.Client(
        api_key="gemini_api_key"
    )
    print("✅ Gemini client initialized with hardcoded API key")
except Exception as e:
    gemini_client = None
    print(f"❌ Failed to init Gemini client: {e}")

# ====== Физические данные ======

PLANET_GRAVITY = {
    "Mercury": 3.7,
    "Venus": 8.87,
    "Earth": 9.81,
    "Moon": 1.62,
    "Mars": 3.71,
    "Jupiter": 24.79,
    "Saturn": 10.44,
    "Uranus": 8.69,
    "Neptune": 11.15,
    "Pluto": 0.62,
}

PLANET_RADIUS = {
    "Mercury": 2_439_700,
    "Venus": 6_051_800,
    "Earth": 6_371_000,
    "Moon": 1_737_100,
    "Mars": 3_389_500,
    "Jupiter": 69_911_000,
    "Saturn": 58_232_000,
    "Uranus": 25_362_000,
    "Neptune": 24_622_000,
    "Pluto": 1_188_300,
}

GAS_GIANTS = {"Jupiter", "Saturn", "Uranus", "Neptune"}

MATERIAL_DENSITIES = {
    'none': 0,
    'uranium': 19100,
    'plutonium': 19800,
    'steel': 7850,
    'gold': 19300,
    'antimony': 6690,
}

MATERIAL_LABELS_RU = {
    'uranium': 'уран',
    'plutonium': 'плутоний',
    'steel': 'сталь',
    'gold': 'золото',
    'antimony': 'сурьма',
}


# ====== Физика ======

def local_g(g0, y, planet_radius_m):
    """g(y) = g0 * (R / (R + y))^2"""
    if not planet_radius_m or planet_radius_m <= 0:
        return g0
    r = planet_radius_m + y
    if r <= 0:
        return g0
    return g0 * (planet_radius_m / r) ** 2


def trajectory_with_drag(v0,
                         alpha_deg,
                         k,
                         rho,
                         g_val,
                         planet_radius_m=None,
                         wind_dir_deg=0.0,
                         wind_speed=0.0,
                         dt=0.001,
                         t_max=3.0):
    """
    Траектория частицы с квадратичным сопротивлением, ветром и g(y).
    Ось X: к унитазу, Y: вверх.
    wind_dir_deg — ОТКУДА дует (0° — от унитаза к тебе).
    """
    if rho <= 0:
        rho = 1.0

    alpha = np.radians(alpha_deg)
    vx = v0 * np.cos(alpha)
    vy = v0 * np.sin(alpha)
    x, y = 0.0, 0.0

    phi = np.radians(wind_dir_deg)
    wx = wind_speed * np.cos(phi + np.pi)
    wy = wind_speed * np.sin(phi + np.pi)

    xs = [x]
    ys = [y]

    for _ in range(int(t_max / dt)):
        g_loc = local_g(g_val, y, planet_radius_m)

        v_rel_x = vx - wx
        v_rel_y = vy - wy
        v_rel = np.hypot(v_rel_x, v_rel_y)

        ax = -k * v_rel_x * v_rel / rho
        ay = -g_loc - k * v_rel_y * v_rel / rho

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        if y < -2.0:
            break

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def generate_particle_trajectories(v0,
                                   alpha_deg,
                                   k,
                                   material,
                                   particle_count,
                                   g_val,
                                   planet_radius_m,
                                   wind_dir_deg,
                                   wind_speed):
    if material == 'none' or particle_count <= 0:
        return []

    rho_p = MATERIAL_DENSITIES.get(material, 0)
    if rho_p <= 0:
        return []

    n_plot = int(min(particle_count, 400))
    disp_coeff = 0.02
    trajectories = []

    for _ in range(n_plot):
        xs, ys = trajectory_with_drag(
            v0=v0,
            alpha_deg=alpha_deg,
            k=k,
            rho=rho_p,
            g_val=g_val,
            planet_radius_m=planet_radius_m,
            wind_dir_deg=wind_dir_deg,
            wind_speed=wind_speed,
            dt=0.001,
            t_max=3.0,
        )
        if len(xs) == 0:
            continue

        sigma = disp_coeff * xs
        ys_disp = ys + np.random.normal(0.0, sigma)
        trajectories.append((xs, ys_disp))

    return trajectories


def simulate_urine_hit_ratio(v0,
                             alpha_deg,
                             k,
                             rho_fluid,
                             g_val,
                             planet_radius_m,
                             wind_dir_deg,
                             wind_speed,
                             d_toilet,
                             rim_height,
                             n_samples=2000):
    if n_samples <= 0:
        return 0.0, 0.0

    hits = 0
    total = n_samples

    angle_sigma_deg = 1.0
    v0_sigma = 0.2
    disp_coeff = 0.01

    for _ in range(n_samples):
        alpha_sample = np.random.normal(alpha_deg, angle_sigma_deg)
        v0_sample = max(0.5, np.random.normal(v0, v0_sigma))

        xs, ys = trajectory_with_drag(
            v0=v0_sample,
            alpha_deg=alpha_sample,
            k=k,
            rho=rho_fluid,
            g_val=g_val,
            planet_radius_m=planet_radius_m,
            wind_dir_deg=wind_dir_deg,
            wind_speed=wind_speed,
            dt=0.001,
            t_max=3.0,
        )
        if len(xs) == 0:
            continue

        sigma = disp_coeff * xs
        ys_disp = ys + np.random.normal(0.0, sigma)

        mask = xs >= d_toilet
        if not np.any(mask):
            continue

        idx = np.argmax(mask)
        y_at = ys_disp[idx]

        if y_at <= rim_height:
            hits += 1

    hit_pct = hits / total * 100.0
    miss_pct = 100.0 - hit_pct
    return hit_pct, miss_pct


def make_plot(v0,
              alpha_deg,
              k,
              rho_fluid,
              material,
              particle_count,
              d_toilet,
              rim_height,
              wind_dir_deg,
              wind_speed,
              g_val,
              planet_name,
              planet_radius_m):
    alpha = np.radians(alpha_deg)
    x_max = max(50.0, d_toilet + 0.5)

    x_par = np.linspace(0, x_max, 800)
    y_par = x_par * np.tan(alpha) - g_val * x_par**2 / (2 * v0**2 * np.cos(alpha)**2)

    x_drag, y_drag = trajectory_with_drag(
        v0=v0,
        alpha_deg=alpha_deg,
        k=k,
        rho=rho_fluid,
        g_val=g_val,
        planet_radius_m=planet_radius_m,
        wind_dir_deg=wind_dir_deg,
        wind_speed=wind_speed,
        dt=0.001,
        t_max=3.0,
    )

    particle_trajs = generate_particle_trajectories(
        v0=v0,
        alpha_deg=alpha_deg,
        k=k,
        material=material,
        particle_count=particle_count,
        g_val=g_val,
        planet_radius_m=planet_radius_m,
        wind_dir_deg=wind_dir_deg,
        wind_speed=wind_speed,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_par, y_par, label='Идеализированная траектория (без сопротивления)')

    if len(x_drag) > 0:
        ax.plot(x_drag, y_drag, linestyle='--',
                label=f'Струя: ρ={rho_fluid:.0f}, k={k:.2f}')

    if particle_trajs:
        for xs, ys in particle_trajs:
            ax.plot(xs, ys, ',', alpha=0.4)
        mat_label = MATERIAL_LABELS_RU.get(material, material)
        ax.plot([], [], ',', alpha=0.7,
                label=f'Тяжёлые частицы: {mat_label}, N={particle_count}')

    ax.axvline(d_toilet, linestyle=':', label=f'Кромка унитаза x={d_toilet:.2f} м')
    ax.axhline(rim_height, linestyle='-.',
               label=f'Высота кромки y={rim_height:.2f} м')

    ax.text(0.02, 0.98,
            f'Планета: {planet_name}, g₀={g_val:.2f} м/с², учитываем g(y)',
            transform=ax.transAxes,
            ha='left', va='top', fontsize=8)

    if planet_name in GAS_GIANTS:
        ax.text(0.02, 0.90,
                'Газовый гигант: гипотетическая платформа в атмосфере',
                transform=ax.transAxes,
                ha='left', va='top', fontsize=7)

    if wind_speed > 0:
        ax.text(0.02, 0.82,
                f'Ветер: {wind_speed:.1f} м/с, {wind_dir_deg:.0f}° (откуда)',
                transform=ax.transAxes,
                ha='left', va='top', fontsize=8)

    ax.set_xlim(0, x_max)
    ax.set_xlabel('Горизонтальное расстояние, м')
    ax.set_ylabel('Высота, м')
    ax.set_title('Траектория струи и частиц (с учётом гравитационного колодца)')
    ax.grid(True)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ====== Текстовые отчёты ======

def build_report_text_formal(planet,
                             g_val,
                             v0,
                             alpha_deg,
                             k,
                             rho_fluid,
                             material,
                             particle_count,
                             d_toilet,
                             rim_height,
                             wind_dir,
                             wind_speed,
                             hit_pct,
                             miss_pct):
    parts = []

    parts.append(
        f"Если на планете {planet} при ускорении свободного падения у поверхности g₀ = {g_val:.2f} м/с² "
        f"струя вылетает со скоростью v₀ = {v0:.1f} м/с под углом {alpha_deg:.0f}° "
        f"при плотности мочи ρ = {rho_fluid:.0f} кг/м³ и коэффициенте сопротивления k = {k:.2f}, "
        f"а расстояние до кромки унитаза составляет {d_toilet:.2f} м, "
        f"кромка расположена на высоте {rim_height:.2f} м относительно уровня члена"
    )

    if wind_speed > 0:
        parts.append(
            f", и дует ветер скоростью {wind_speed:.1f} м/с под углом {wind_dir:.0f}° (откуда дует)"
        )
    else:
        parts.append(", при отсутствии ветра")

    parts.append(
        f", то с учётом изменения g(y) модель показывает, что примерно {hit_pct:.1f}% объёма струи попадает в унитаз, "
        f"а около {miss_pct:.1f}% пролетают мимо."
    )

    if planet in GAS_GIANTS:
        parts.append(
            " Для газового гиганта расчёт интерпретируется как струя с условной платформы "
            "в атмосфере на уровне, соответствующем указанному g; твёрдой поверхности в реальности нет."
        )

    if material != 'none' and particle_count > 0:
        mat_label = MATERIAL_LABELS_RU.get(material, material)
        rho_mat = MATERIAL_DENSITIES.get(material, 0)
        parts.append(
            f" В модель включены {particle_count} тяжёлых частиц из материала «{mat_label}» "
            f"(ρ ≈ {rho_mat} кг/м³), что увеличивает разброс и риск загрязнения окружающего пространства."
        )

    return "".join(parts)


def build_report_text_gopnik_fallback(formal_text: str) -> str:
    return (
        "Братан, смотри какая тема.\n\n"
        + formal_text
        .replace("Если на планете", "Короче, локация —")
        .replace("струя вылетает", "ты заряжаешь струю")
        .replace("модель показывает, что примерно", "по расчётам выходит примерно")
        .replace("объёма струи попадает в унитаз", "летит куда надо, в парашу")
        .replace("пролетают мимо", "улетают мимо кассы")
        .replace("расчёт интерпретируется как", "это мы считаем как")
        .replace("что увеличивает разброс и риск загрязнения окружающего пространства",
                 "так что разброс только растёт, а уборка превращается в квест")
    )


def build_report_text_gopnik_via_gemini(formal_text: str) -> str:
    """
    Генерация пацанского отчёта через Gemini.
    Логируем всё важное в консоль, чтобы дебажить поведение.
    """
    if gemini_client is None:
        print("[Gemini] Client is None, using fallback.")
        return build_report_text_gopnik_fallback(formal_text)

    run_id = f"run_{int(time.time() * 1000)}"

    prompt = (
        "Перепиши следующий отчёт на русском языке в пацанском, разговорном стиле: "
        "саркастично, злобно, как будто рассказываешь это не очень лицеприятному человеку. "
        "Числа, проценты, планеты, материалы облагородь обильным матерным языком. "
        "Не меняй значения, но докинь позволительных грубостей."
        "Выведи готовый текст. Не присылай run id\n\n"
        f"run_id: {run_id}\n"
        f"{formal_text}"
    )

    try:
        print("\n" + "=" * 40)
        print(f"[Gemini] REQUEST ({run_id}) prompt:")
        print(prompt)
        print("=" * 40)

        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        # Логируем сырой объект ответа (может быть большим, но для дебага норм)
        print(f"[Gemini] RAW RESPONSE ({run_id}): {resp!r}")

        text = (getattr(resp, "text", "") or "").strip()

        print(f"[Gemini] PARSED TEXT ({run_id}): {repr(text)}")

        if not text:
            print(f"[Gemini] EMPTY TEXT ({run_id}), using fallback.")
            return build_report_text_gopnik_fallback(formal_text)

        print(f"[Gemini] OK ({run_id}) using Gemini text.")
        return text

    except Exception as e:
        print(f"[Gemini] ERROR ({run_id}): {repr(e)}")
        print("[Gemini] Using fallback text.")
        return build_report_text_gopnik_fallback(formal_text)



def build_report_text(style,
                      use_gemini,
                      **kwargs):
    formal = build_report_text_formal(**kwargs)

    if style == "gopnik":
        if use_gemini:
            return build_report_text_gopnik_via_gemini(formal)
        else:
            return build_report_text_gopnik_fallback(formal)

    return formal


# ====== Flask route ======

@app.route('/', methods=['GET', 'POST'])
def index():
    # Дефолты
    v0 = 3.0
    alpha_deg = 45.0
    k = 0.2
    rho_fluid = 1000.0

    material = 'none'
    particle_count = 0

    d_toilet = 1.0
    rim_height = -0.4

    wind_dir = 0.0
    wind_speed = 0.0

    planet = "Earth"
    report_style = "formal"
    use_gemini = False

    if request.method == 'POST':
        try:
            v0 = float(request.form.get('v0', v0))
            alpha_deg = float(request.form.get('alpha', alpha_deg))
            k = float(request.form.get('k', k))
            rho_fluid = float(request.form.get('rho', rho_fluid))

            material = request.form.get('material', material)
            particle_count = int(request.form.get('particle_count', particle_count))

            d_toilet = float(request.form.get('d_toilet', d_toilet))
            rim_height = float(request.form.get('rim_height', rim_height))

            wind_dir = float(request.form.get('wind_dir', wind_dir))
            wind_speed = float(request.form.get('wind_speed', wind_speed))

            planet = request.form.get('planet', planet)
            report_style = request.form.get('report_style', report_style)
            use_gemini = (request.form.get('use_gemini') == 'on')
        except ValueError:
            pass

    g_val = PLANET_GRAVITY.get(planet, PLANET_GRAVITY["Earth"])
    planet_radius_m = PLANET_RADIUS.get(planet)

    plot_data = make_plot(
        v0=v0,
        alpha_deg=alpha_deg,
        k=k,
        rho_fluid=rho_fluid,
        material=material,
        particle_count=particle_count,
        d_toilet=d_toilet,
        rim_height=rim_height,
        wind_dir_deg=wind_dir,
        wind_speed=wind_speed,
        g_val=g_val,
        planet_name=planet,
        planet_radius_m=planet_radius_m,
    )

    hit_pct, miss_pct = simulate_urine_hit_ratio(
        v0=v0,
        alpha_deg=alpha_deg,
        k=k,
        rho_fluid=rho_fluid,
        g_val=g_val,
        planet_radius_m=planet_radius_m,
        wind_dir_deg=wind_dir,
        wind_speed=wind_speed,
        d_toilet=d_toilet,
        rim_height=rim_height,
        n_samples=2000,
    )

    report_text = build_report_text(
        style=report_style,
        use_gemini=use_gemini,
        planet=planet,
        g_val=g_val,
        v0=v0,
        alpha_deg=alpha_deg,
        k=k,
        rho_fluid=rho_fluid,
        material=material,
        particle_count=particle_count,
        d_toilet=d_toilet,
        rim_height=rim_height,
        wind_dir=wind_dir,
        wind_speed=wind_speed,
        hit_pct=hit_pct,
        miss_pct=miss_pct,
    )

    return render_template(
        'index.html',
        plot_data=plot_data,
        v0=v0,
        alpha_deg=alpha_deg,
        k=k,
        rho=rho_fluid,
        material=material,
        particle_count=particle_count,
        d_toilet=d_toilet,
        rim_height=rim_height,
        wind_dir=wind_dir,
        wind_speed=wind_speed,
        hit_pct=hit_pct,
        miss_pct=miss_pct,
        planet=planet,
        planet_gravity=PLANET_GRAVITY,
        report_text=report_text,
        report_style=report_style,
        use_gemini=use_gemini,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
