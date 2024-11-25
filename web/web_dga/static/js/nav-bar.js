// 获取所有的导航链接  
var navLinks = document.querySelectorAll('.nav-bar li a');  
  
// 遍历所有的导航链接，并为当前页面的链接添加active类  
navLinks.forEach(function(link) {  
    if (link.href === window.location.href) {  
        link.classList.add('active');  
    }  
});