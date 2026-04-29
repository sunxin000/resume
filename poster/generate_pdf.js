const puppeteer = require('puppeteer-core');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  
  const htmlPath = path.resolve(__dirname, 'index.html');
  await page.goto('file://' + htmlPath, { waitUntil: 'networkidle0', timeout: 30000 });
  
  // Wait for fonts and KaTeX to load
  await new Promise(r => setTimeout(r, 3000));
  
  // A0 landscape: 1189mm x 841mm
  await page.pdf({
    path: path.resolve(__dirname, 'TTARAG_ICASSP2026_Poster.pdf'),
    width: '1189mm',
    height: '841mm',
    printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    preferCSSPageSize: true
  });
  
  console.log('PDF generated: TTARAG_ICASSP2026_Poster.pdf');
  await browser.close();
})();
