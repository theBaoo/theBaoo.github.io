// scripts/inject-custom-css.js
hexo.extend.filter.register('after_render:html', function (str, data) {
  const linkTag = '<link rel="stylesheet" href="/css/custom.css">';
  return str.replace(/<head>/, `<head>\n  ${linkTag}`);
});