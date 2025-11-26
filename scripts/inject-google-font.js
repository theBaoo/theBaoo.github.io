// scripts/inject-google-font.js
hexo.extend.filter.register('after_render:html', function (str, data) {
  const fontLink = `<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap" rel="stylesheet">`;
  return str.replace(/<head>/, `<head>\n  ${fontLink}`);
});