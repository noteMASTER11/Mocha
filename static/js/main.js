document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('params-form');
    const loader = document.getElementById('loader-overlay');

    if (!form || !loader) return;

    form.addEventListener('submit', function () {
        const reportStyleSelect = form.querySelector('#reportStyleSelect');
        const useGeminiCheckbox = form.querySelector('#useGemini');
        const geminiLine = loader.querySelector('.loader-step-gemini');

        form.classList.add('locked');

        // Управляем строкой про Gemini
        const useGemini =
            reportStyleSelect &&
            reportStyleSelect.value === 'gopnik' &&
            useGeminiCheckbox &&
            useGeminiCheckbox.checked;

        if (geminiLine) {
            if (useGemini) {
                geminiLine.classList.remove('d-none');
            } else {
                geminiLine.classList.add('d-none');
            }
        }

        loader.classList.remove('hidden');
    });
});
