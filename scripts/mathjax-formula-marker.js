'use strict';

const FORMULA = /\$\$[\s\S]+?\$\$|(?<!\\)\$(?!\$)(?:\\.|[^$\r\n])+?(?<!\\)\$(?!\$)|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)/;
const NON_PROSE = /<(pre|code|script|style)\b[\s\S]*?<\/\1>/gi;
const ARTICLE = /<article\b[^>]*class="[^"]*\bmd-text\b[^"]*"[^>]*>[\s\S]*?<\/article>/g;

function containsFormula(html) {
  return FORMULA.test(html.replace(NON_PROSE, ''));
}

hexo.extend.filter.register('after_render:html', html => html.replace(ARTICLE, article => {
  if (!containsFormula(article)) return article;

  // Stellar mounts MathJax against the activation element itself, so mark the
  // final article DOM. Doing this after the full page render keeps generate and
  // server routes consistent, unlike wrapping cached post content separately.
  return article.replace(
    /<article\b[^>]*class="([^"]*\bmd-text\b[^"]*)"/,
    '<article class="$1 has-jax"'
  );
}), 50);
