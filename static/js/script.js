// ── Image Preview on Upload Page ─────────────────────────────
function previewImage(event) {
    const file = event.target.files[0];
    const container = document.getElementById('preview-container');
    const preview = document.getElementById('image-preview');
    const dropArea = document.getElementById('dropArea');

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            container.style.display = 'block';
            if (dropArea) {
                dropArea.style.borderColor = '#00b4d8';
                dropArea.style.background = 'rgba(0,180,216,0.03)';
                dropArea.querySelector('p').textContent = file.name;
            }
        };
        reader.readAsDataURL(file);
    } else {
        container.style.display = 'none';
    }
}

// ── Drag & Drop support ─────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
    const dropArea = document.getElementById('dropArea');
    if (dropArea) {
        const fileInput = document.getElementById('mri_image');

        ['dragenter', 'dragover'].forEach(function (evt) {
            dropArea.addEventListener(evt, function (e) {
                e.preventDefault();
                dropArea.style.borderColor = '#00b4d8';
                dropArea.style.background = 'rgba(0,180,216,0.06)';
            });
        });

        ['dragleave', 'drop'].forEach(function (evt) {
            dropArea.addEventListener(evt, function (e) {
                e.preventDefault();
                dropArea.style.borderColor = '';
                dropArea.style.background = '';
            });
        });

        dropArea.addEventListener('drop', function (e) {
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                previewImage({ target: fileInput });
            }
        });
    }

    // ── Loading overlay on form submit ──────────────────────
    var form = document.getElementById('diagnosisForm');
    var overlay = document.getElementById('loadingOverlay');
    if (form && overlay) {
        form.addEventListener('submit', function () {
            overlay.classList.add('active');
        });
    }
});

// ── Reports Page: Search & Filter ───────────────────────────
function filterTable() {
    const searchInput = document.getElementById('searchInput');
    const filterSelect = document.getElementById('filterSelect');

    if (!searchInput || !filterSelect) return;

    const searchTerm = searchInput.value.toLowerCase();
    const filterValue = filterSelect.value;
    const table = document.getElementById('reportsTable');
    const rows = table.querySelectorAll('tbody tr');

    rows.forEach(function (row) {
        const name = row.cells[1] ? row.cells[1].textContent.toLowerCase() : '';
        const result = row.getAttribute('data-result') || '';

        const matchesSearch = name.includes(searchTerm);
        const matchesFilter = filterValue === 'all' || result === filterValue;

        row.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
    });
}
