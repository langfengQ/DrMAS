import katex from 'katex';

// Only explicit math elements are transformed; dollar amounts in prose stay intact.
export function renderMath(html) {
  return html.replace(/<(div|span)\b([^>]*\bdata-math\b[^>]*)>([\s\S]*?)<\/\1>/g, (_, tag, attributes, content) => {
    const source = content.trim();
    const display = source.startsWith('$$') && source.endsWith('$$');
    const inline = source.startsWith('\\(') && source.endsWith('\\)');
    if (!display && !inline) throw new Error('Math must use $$...$$ or \\(...\\) delimiters');
    const rendered = katex.renderToString(source.slice(2, -2).trim(), {
      displayMode: display,
      output: 'htmlAndMathml',
      throwOnError: true,
      strict: 'error',
      trust: false,
      macros: { '\\bm': '\\boldsymbol{#1}' },
    });
    return `<${tag}${attributes}>${rendered}</${tag}>`;
  });
}
