'use strict';

const { escapeHTML } = require('hexo-util');

function categoryRoot(category) {
  let current = category;
  const seen = new Set();

  while (current?.parent && !seen.has(current._id)) {
    seen.add(current._id);
    const parentId = typeof current.parent === 'object' ? current.parent._id : current.parent;
    const parent = parentId ? hexo.model('Category').findById(parentId) : null;
    if (!parent) break;
    current = parent;
  }

  return current;
}

hexo.extend.tag.register('series_index', function () {
  const root = String(hexo.config.root || '/').replace(/\/?$/, '/');
  const urlFor = path => {
    const value = String(path || '').replace(/^\/+/, '');
    return /^(?:https?:)?\/\//.test(value) ? value : `${root}${value}`;
  };
  const formatDate = value => value?.format?.('YYYY-MM-DD') || '';
  const groups = new Map();

  hexo.locals.get('posts').forEach(post => {
    if (post.published === false || post.hide === true || !post.title) return;

    const roots = new Map();
    const categories = post.categories?.toArray?.() || [];

    for (const category of categories) {
      const root = categoryRoot(category);
      if (root) roots.set(root._id, root);
    }

    if (roots.size === 0) {
      roots.set('__uncategorized__', {
        _id: '__uncategorized__',
        name: '未分类',
        path: hexo.config.category_dir
      });
    }

    for (const root of roots.values()) {
      if (!groups.has(root._id)) {
        groups.set(root._id, {
          category: root,
          posts: []
        });
      }
      groups.get(root._id).posts.push(post);
    }
  });

  const ordered = [...groups.values()].map(group => {
    group.posts.sort((a, b) => b.date.valueOf() - a.date.valueOf());
    return {
      ...group,
      latest: group.posts[0]?.date?.valueOf?.() || 0
    };
  }).sort((a, b) => b.latest - a.latest || a.category.name.localeCompare(b.category.name, 'zh-CN'));

  let html = '<div class="post-list archives series-index">';

  for (const group of ordered) {
    const category = group.category;
    const categoryHref = urlFor('/' + String(category.path || '').replace(/^\/+/, ''));

    html += '<article class="series-group">';
    html += '<div class="series-header">';
    html += `<a class="series-title" href="${escapeHTML(categoryHref)}">${escapeHTML(category.name)}</a>`;
    html += `<span class="series-count">${group.posts.length} 篇</span>`;
    html += '</div>';

    for (const post of group.posts) {
      const href = urlFor(post.path);
      const date = formatDate(post.date, 'YYYY-MM-DD');
      html += '<div class="archive-list">';
      html += `<a class="post fs14" href="${escapeHTML(href)}">`;
      html += `<time>${escapeHTML(date)}</time>`;
      html += `<span>${escapeHTML(post.title)}</span>`;
      html += '</a>';
      html += '</div>';
    }

    html += '</article>';
  }

  html += '</div>';
  return html;
});

hexo.extend.filter.register('after_render:html', html => {
  if (!html.includes('class="post-list archives series-index"')) return html;

  const root = String(hexo.config.root || '/').replace(/\/?$/, '/');
  const urlFor = path => {
    const value = String(path || '').replace(/^\/+/, '');
    return /^(?:https?:)?\/\//.test(value) ? value : `${root}${value}`;
  };
  const nav = '<div class="listing-nav"><div class="listing-nav__surface"><div class="listing-nav__viewport"><nav class="post">'
    + `<a href="${root}">近期发布</a>`
    + `<a href="${urlFor(hexo.config.category_dir)}">分类</a>`
    + `<a href="${urlFor(hexo.config.tag_dir)}">标签</a>`
    + `<a class="active" href="${urlFor('series/')}">系列</a>`
    + `<a href="${urlFor(hexo.config.archive_dir)}">归档</a>`
    + '</nav></div></div></div>';

  return html.replace(
    /(<article\b[^>]*class="[^"]*\bmd-text\b[^"]*"[^>]*>)(?=\s*<div class="post-list archives series-index">)/,
    `${nav}$1`
  );
}, 55);

hexo.extend.filter.register('after_render:html', html => {
  if (!html.includes('listing-nav__viewport')) return html;

  return html.replace(
    /(<div class="listing-nav__viewport"><nav\b[^>]*>)([\s\S]*?)(<\/nav><\/div><\/div><\/div>)/g,
    (match, start, links, end) => {
      const archive = /<a\b[^>]*>\s*归档\s*<\/a>/;
      const series = /<a\b[^>]*>\s*系列\s*<\/a>/;
      if (!archive.test(links) || !series.test(links)) return match;

      return `${start}${links.replace(
        /(<a\b[^>]*>\s*归档\s*<\/a>)(\s*)(<a\b[^>]*>\s*系列\s*<\/a>)/,
        '$3$2$1'
      )}${end}`;
    }
  );
}, 60);
