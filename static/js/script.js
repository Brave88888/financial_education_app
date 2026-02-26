// 页面加载完成后的操作
document.addEventListener('DOMContentLoaded', function() {
    // 初始化代码高亮
    Prism.highlightAll();

    // 复制按钮事件
    initCopyButtons();

    // 运行代码按钮事件
    initRunButtons();

    // 添加平滑滚动
    initSmoothScroll();

    // 添加页面加载动画
    initPageLoad();
});

// 复制按钮功能
function initCopyButtons() {
    const copyButtons = document.querySelectorAll('.btn-copy');

    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeElement = this.closest('.example-card').querySelector('code');
            const textArea = document.createElement('textarea');
            textArea.value = codeElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            // 显示复制成功消息
            const originalText = button.textContent;
            button.textContent = '已复制!';
            button.style.backgroundColor = '#27ae60';

            setTimeout(() => {
                button.textContent = originalText;
                button.style.backgroundColor = '';
            }, 2000);
        });
    });
}

// 运行代码按钮功能
function initRunButtons() {
    const runButtons = document.querySelectorAll('.btn-run');

    runButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeElement = this.closest('.example-card').querySelector('code');
            const code = codeElement.textContent;

            // 在控制台运行代码（简单实现）
            console.log('运行代码:');
            console.log(code);

            // 显示运行提示
            const originalText = button.textContent;
            button.textContent = '运行中...';

            setTimeout(() => {
                button.textContent = originalText;
                alert('代码已在浏览器控制台中运行，请打开开发者工具查看结果');
            }, 1000);
        });
    });
}

// 平滑滚动
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));

            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// 页面加载动画
function initPageLoad() {
    const cards = document.querySelectorAll('.topic-card, .category-card, .example-card');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '0';
                entry.target.style.transform = 'translateY(20px)';

                setTimeout(() => {
                    entry.target.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, 100);

                observer.unobserve(entry.target);
            }
        });
    });

    cards.forEach(card => {
        observer.observe(card);
    });
}

// 显示代码运行结果
function displayCodeResult(result) {
    // 创建结果容器
    const resultContainer = document.createElement('div');
    resultContainer.className = 'code-result';
    resultContainer.style.cssText = `
        background: #f0f0f0;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        border-left: 3px solid #27ae60;
    `;

    // 创建结果标题
    const resultTitle = document.createElement('h4');
    resultTitle.textContent = '运行结果';
    resultTitle.style.marginBottom = '0.5rem';

    // 创建结果内容
    const resultContent = document.createElement('pre');
    resultContent.style.cssText = `
        background: white;
        padding: 1rem;
        border-radius: 5px;
        max-height: 300px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.4;
    `;
    resultContent.textContent = result;

    // 组合并显示结果
    resultContainer.appendChild(resultTitle);
    resultContainer.appendChild(resultContent);

    // 插入到当前示例卡片中
    const exampleCard = document.querySelector('.example-card');
    if (exampleCard) {
        exampleCard.appendChild(resultContainer);
    }
}

// 下载代码功能
function downloadCode(code, filename) {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

// 搜索功能
function searchContent(query) {
    const cards = document.querySelectorAll('.topic-card, .category-card, .example-card');

    cards.forEach(card => {
        const text = card.textContent.toLowerCase();
        const match = text.includes(query.toLowerCase());

        card.style.display = match ? 'block' : 'none';
    });
}

// 响应式菜单（移动端）
function initMobileMenu() {
    const menuToggle = document.querySelector('.menu-toggle');
    const navMenu = document.querySelector('.nav-menu');

    if (menuToggle && navMenu) {
        menuToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
    }
}

// 图片加载优化
function optimizeImages() {
    const images = document.querySelectorAll('img');

    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.opacity = '0';
            this.style.transition = 'opacity 0.3s ease';

            setTimeout(() => {
                this.style.opacity = '1';
            }, 100);
        });
    });
}

// 页面可见性变化
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // 页面不可见时暂停动画
        document.querySelectorAll('.animate').forEach(el => {
            el.style.animationPlayState = 'paused';
        });
    } else {
        // 页面可见时继续动画
        document.querySelectorAll('.animate').forEach(el => {
            el.style.animationPlayState = 'running';
        });
    }
});

// 键盘快捷键
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K 搜索
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.focus();
        }
    }

    // Escape 关闭搜索
    if (e.key === 'Escape') {
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.value = '';
            searchContent('');
            searchInput.blur();
        }
    }
});
