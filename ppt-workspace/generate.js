const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require('/Users/sunxin/.codefuse/engine/cc/skills/pptx/scripts/html2pptx');

async function main() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'Xin Sun';
    pptx.title = 'Interview Presentation - Xin Sun';

    const slides = [
        'slide01-title.html',              // 1. Title
        'slide02-about.html',              // 2. About me
        'slide04-publications.html',       // 3. Publications overview
        'slide04-dta-method.html',         // 4. DTA: problem + method
        'slide05-dta-results.html',        // 5. DTA: framework + results
        'slide06-ttarag.html',             // 6. TTARAG overview
        'slide05-kbqa-motivation.html',    // 7. KBQA: motivation
        'slide06-kbqa-overview.html',      // 8. KBQA: framework
        'slide07-kbqa-actions.html',       // 9. KBQA: action space
        'slide08-kbqa-pipeline.html',      // 10. KBQA: 2-stage pipeline
        'slide09-kbqa-rrs.html',           // 11. KBQA: RRS
        'slide10-kbqa-grpo.html',          // 12. KBQA: GRPO
        'slide11-kbqa-rrcg.html',          // 13. KBQA: RRCG
        'slide12-kbqa-results.html',       // 14. KBQA: results
        'slide13-others-intern.html',      // 15. Other works + internship
        'slide13-future.html',             // 16. Future plans
        'slide14-thanks.html',             // 17. Thank you
    ];

    for (const slideFile of slides) {
        const htmlPath = path.join(__dirname, 'slides', slideFile);
        console.log(`Processing: ${slideFile}`);
        await html2pptx(htmlPath, pptx);
    }

    const outputPath = path.join(__dirname, '..', 'interview-presentation.pptx');
    await pptx.writeFile({ fileName: outputPath });
    console.log(`\nDone! ${slides.length} slides. Saved to: ${outputPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
