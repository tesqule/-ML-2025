let selectedFile = null;

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Проверка типа файла
                const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
                const fileType = file.type.toLowerCase();

                if (!validTypes.includes(fileType)) {
                    alert('Пожалуйста, выберите изображение в формате JPG, PNG, WEBP или GIF');
                    fileInput.value = '';
                    return;
                }

                // Проверка размера файла (макс 15MB)
                const maxSize = 15 * 1024 * 1024; // 15MB
                if (file.size > maxSize) {
                    alert('Файл слишком большой. Максимальный размер: 15MB');
                    fileInput.value = '';
                    return;
                }

                selectedFile = file;

                // Показываем превью
                const previewContainer = document.getElementById('preview-container');
                const previewImage = document.getElementById('preview-image');
                const reader = new FileReader();

                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewContainer.style.display = 'block';
                }

                reader.onerror = function() {
                    alert('Ошибка при чтении файла');
                    previewContainer.style.display = 'none';
                    selectedFile = null;
                    fileInput.value = '';
                }

                reader.readAsDataURL(file);
            }
        });
    }
});

function uploadImage() {
    if (!selectedFile) {
        showNotification('Пожалуйста, выберите файл', 'error');
        return;
    }

    const loading = document.getElementById('loading');
    const previewContainer = document.getElementById('preview-container');

    loading.style.display = 'block';
    previewContainer.style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Показываем уведомление о начале обработки
    showNotification('Начинаю обработку изображения...', 'info');

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        loading.style.display = 'none';

        if (data.success) {
            console.log('Результаты успешно получены:', data.results);
            showNotification('Изображение успешно обработано!', 'success');

            // Сохраняем результаты в localStorage
            localStorage.setItem('detection_results', JSON.stringify(data.results));

            // Небольшая задержка перед переходом
            setTimeout(() => {
                window.location.href = '/result';
            }, 800);

        } else {
            console.error('Ошибка от сервера:', data.error);
            showNotification('Ошибка: ' + (data.error || 'Неизвестная ошибка'), 'error');
            previewContainer.style.display = 'block';
        }
    })
    .catch(error => {
        loading.style.display = 'none';
        previewContainer.style.display = 'block';
        console.error('Ошибка при загрузке файла:', error);
        showNotification('Ошибка сети или сервера: ' + error.message, 'error');
    });
}

function cancelUpload() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('preview-container').style.display = 'none';
    document.getElementById('preview-image').src = '';
}

// Функция для показа уведомлений
function showNotification(message, type = 'info') {
    // Удаляем предыдущее уведомление если есть
    const oldNotification = document.querySelector('.notification');
    if (oldNotification) {
        oldNotification.remove();
    }

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        animation: slideIn 0.3s ease;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    `;

    if (type === 'success') {
        notification.style.background = '#28a745';
    } else if (type === 'error') {
        notification.style.background = '#dc3545';
    } else {
        notification.style.background = '#007bff';
    }

    document.body.appendChild(notification);

    // Автоматическое скрытие через 4 секунды
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Добавляем стили для анимации уведомлений
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }

    .notification {
        font-family: Arial, sans-serif;
    }
`;
document.head.appendChild(style);

// Функция для отображения размера файла
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}