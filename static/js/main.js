document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');

    const resultArea = document.getElementById('result-area');
    const resultTitle = document.getElementById('result-title');
    const resultContent = document.getElementById('result-content');
    const resultSuggestion = document.getElementById('result-suggestion');

    const btnPredictCombined = document.getElementById('btn-predict-combined');
    const detailsForm = document.getElementById('details-form');

    const fileName = document.getElementById('file-name');
    const deleteBtn = document.getElementById('delete-image-btn');
    const uploadText = document.getElementById('upload-text');
    const uploadSubtext = document.getElementById('upload-subtext');

    /* ---------------- MODAL ---------------- */
    const rangeModal = document.getElementById('range-modal');
    const btnShowRanges = document.getElementById('btn-show-ranges');
    const closeModal = document.getElementById('close-modal');

    btnShowRanges.addEventListener('click', () => {
        rangeModal.style.display = 'flex';
    });

    closeModal.addEventListener('click', () => {
        rangeModal.style.display = 'none';
    });

    rangeModal.addEventListener('click', (e) => {
        if (e.target === rangeModal) {
            rangeModal.style.display = 'none';
        }
    });

    /* ---------------- IMAGE UPLOAD ---------------- */
    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileName.textContent = `Selected: ${fileInput.files[0].name}`;
            fileName.style.display = 'block';

            uploadText.style.display = 'none';
            uploadSubtext.style.display = 'none';

            deleteBtn.style.display = 'inline-block';
        }
    });

    deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();

        fileInput.value = '';
        fileName.style.display = 'none';

        uploadText.style.display = 'block';
        uploadSubtext.style.display = 'block';

        deleteBtn.style.display = 'none';
        resultArea.style.display = 'none';
    });

    /* ---------------- PREDICTION ---------------- */
    btnPredictCombined.addEventListener('click', async () => {
        if (fileInput.files.length === 0) {
            alert('Please upload a blood smear image!');
            return;
        }

        const formData = new FormData(detailsForm);

        const requiredFields = ["wbc", "rbc", "platelets", "hb", "blasts", "age"];
        for (let field of requiredFields) {
            const value = formData.get(field);
            if (!value || value.trim() === "") {
                alert(`Please enter ${field.toUpperCase()} value`);
                return;
            }
        }

        formData.append('file', fileInput.files[0]);

        resultArea.style.display = 'block';
        resultTitle.innerText = 'Predicted Blood Cancer Type';
        resultContent.innerText = 'Analyzing image and clinical parameters...';
        resultSuggestion.innerText = '';

        try {
            const response = await fetch('/predict_combined', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status === 'success') {
                resultContent.innerText = result.prediction;

                if (result.suggestion) {
                    resultSuggestion.innerText =
                        "Medical Suggestion: " + result.suggestion;
                }
            } else {
                resultContent.innerText = 'Error: ' + result.message;
            }
        } catch (e) {
            resultContent.innerText = 'Error processing request';
        }
    });
});