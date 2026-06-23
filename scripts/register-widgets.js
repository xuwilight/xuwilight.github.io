'use strict';
const path = require('path');
const fs = require('fs');

// Register custom widget views into the theme so partial() can find them
hexo.on('generateBefore', function() {
  const theme = hexo.theme;
  const srcDir = path.join(hexo.base_dir, 'layout', '_partial', 'widgets');

  // Register each custom widget
  const widgets = ['categories'];

  widgets.forEach(name => {
    const filePath = path.join(srcDir, name + '.ejs');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf-8');
      theme.setView('_partial/widgets/' + name + '.ejs', content);
      console.log('[custom-widgets] Registered:', name);
    }
  });
});
