/* Card content registry for TTARAG poster */
const CARD_REGISTRY = {
  tldr: {
    title: 'TL;DR',
    color: null,
    grow: false,
    body: React.createElement(React.Fragment, null,
      React.createElement('div', {className:'hl'}, React.createElement('p', null, 'We propose TTARAG, a test-time adaptation method that dynamically updates LLM parameters during inference by learning to predict retrieved content, enabling automatic domain adaptation for RAG systems without labeled data.')),
      React.createElement('ul', null,
        React.createElement('li', null, React.createElement('b', null, 'Self-supervised adaptation: '), 'Split retrieved passages into prefix-suffix pairs for prediction-based parameter updates'),
        React.createElement('li', null, React.createElement('b', null, 'No labeled data needed: '), 'Fully unsupervised — adapts using only retrieved passages at test time'),
        React.createElement('li', null, React.createElement('b', null, 'Consistent improvements: '), 'Best results in 19/24 settings across 6 specialized domains, up to +25% on medical QA')
      )
    ),
  },
  motivation: {
    title: 'Motivation',
    color: 'orange',
    grow: false,
    body: React.createElement(React.Fragment, null,
      React.createElement('ul', null,
        React.createElement('li', null, 'RAG systems struggle with ', React.createElement('b', null, 'distribution shifts'), ' when adapting to specialized domains (medical, finance, etc.)'),
        React.createElement('li', null, 'Standard RAG provides retrieved passages but the LLM may not ', React.createElement('b', null, 'effectively utilize domain-specific context')),
        React.createElement('li', null, 'Pre-trained RAG models (Ret-Robust, RAAT, Self-RAG) show ', React.createElement('b', null, 'limited generalization'), ' — often worse than vanilla models'),
        React.createElement('li', null, React.createElement('b', null, 'Test-time adaptation (TTA)'), ' enables dynamic parameter updates at inference without new labeled data')
      )
    ),
  },
  framework: {
    title: 'Framework Overview',
    color: 'teal',
    grow: true,
    body: React.createElement(React.Fragment, null,
      React.createElement('div', {className:'fig'}, React.createElement('div', {className:'fig-wrap'}, React.createElement('img', {src:'framework.png', alt:'TTARAG Framework'})),
        React.createElement('div', {className:'cap'}, React.createElement('b', null, 'Figure 1. '), 'Comparison between Standard RAG and TTARAG. TTARAG introduces a test-time adaptation module that splits passages into prefix-suffix pairs, performs self-supervised learning, and updates model parameters before generation.')
      )
    ),
  },
  method: {
    title: 'Methodology',
    color: 'teal',
    grow: false,
    body: React.createElement(React.Fragment, null,
      React.createElement('div', {className:'eq'}, '$\\mathcal{L}_{adapt} = -\\sum_{i=1}^{k} \\log P(p_i^{suffix} | p_i^{prefix}, q; \\theta)$'),
      React.createElement('ul', null,
        React.createElement('li', null, React.createElement('b', null, 'Context Processing: '), 'Filter short passages, then split each into prefix-suffix pairs at natural linguistic boundaries (punctuation)'),
        React.createElement('li', null, React.createElement('b', null, 'Prefix-Suffix Prediction: '), 'Train model to predict suffix given prefix + query — learns domain-specific language patterns'),
        React.createElement('li', null, React.createElement('b', null, 'Parameter Update: '), 'AdamW optimizer with gradient accumulation (2 steps) and clipping; only 3 pairs needed'),
        React.createElement('li', null, React.createElement('b', null, 'Response Generation: '), 'Generate final answer using adapted parameters $\\theta\'$')
      ),
      React.createElement('div', {className:'eq'}, '$y = \\arg\\max_y P(y | q, \\{p_1, ..., p_k\\}; \\theta\')$')
    ),
  },
  results_table: {
    title: 'Main Results — Accuracy (%)',
    color: 'purple',
    grow: false,
    body: React.createElement(React.Fragment, null,
      React.createElement('table', null,
        React.createElement('thead', null,
          React.createElement('tr', null,
            React.createElement('th', null, 'Model'),
            React.createElement('th', null, 'Finance'),
            React.createElement('th', null, 'Sports'),
            React.createElement('th', null, 'Music'),
            React.createElement('th', null, 'Movie'),
            React.createElement('th', null, 'Open'),
            React.createElement('th', null, 'Overall'),
            React.createElement('th', null, 'BioASQ'),
            React.createElement('th', null, 'PubMed')
          )
        ),
        React.createElement('tbody', null,
          React.createElement('tr', null, React.createElement('td', {colSpan:9, style:{fontWeight:700, fontStyle:'italic', background:'#f0f0f0'}}, 'Llama-3.1-8b-it')),
          React.createElement('tr', null, React.createElement('td',null,'naive-rag'), React.createElement('td',null,'17.4'), React.createElement('td',null,'27.6'), React.createElement('td',null,'34.9'), React.createElement('td',null,'31.3'), React.createElement('td',null,'42.4'), React.createElement('td',null,'29.8'), React.createElement('td',null,'55.6'), React.createElement('td',null,'46.6')),
          React.createElement('tr', null, React.createElement('td',null,'+CoT'), React.createElement('td',null,'17.9'), React.createElement('td',null,'30.2'), React.createElement('td',null,'37.6'), React.createElement('td',null,'31.5'), React.createElement('td',null,'45.8'), React.createElement('td',null,'31.6'), React.createElement('td',null,'54.6'), React.createElement('td',null,'50.8')),
          React.createElement('tr', {className:'our-row'}, React.createElement('td',null,'TTARAG'), React.createElement('td',{className:'best'},'20.1'), React.createElement('td',null,'29.5'), React.createElement('td',{className:'best'},'37.7'), React.createElement('td',{className:'best'},'34.6'), React.createElement('td',null,'41.5'), React.createElement('td',{className:'best'},'31.9'), React.createElement('td',{className:'best'},'75.0'), React.createElement('td',{className:'best'},'57.4')),
          React.createElement('tr', {style:{background:'#e8f5e9'}}, React.createElement('td',null,'Δ vs naive'), React.createElement('td',{className:'delta-pos'},'+2.7'), React.createElement('td',{className:'delta-pos'},'+1.9'), React.createElement('td',{className:'delta-pos'},'+2.8'), React.createElement('td',{className:'delta-pos'},'+3.3'), React.createElement('td',{className:'delta-neg'},'-0.9'), React.createElement('td',{className:'delta-pos'},'+2.1'), React.createElement('td',{className:'delta-pos'},'+19.4'), React.createElement('td',{className:'delta-pos'},'+10.8')),
          React.createElement('tr', null, React.createElement('td', {colSpan:9, style:{fontWeight:700, fontStyle:'italic', background:'#f0f0f0'}}, 'Llama-2-7b-chat')),
          React.createElement('tr', null, React.createElement('td',null,'naive-rag'), React.createElement('td',null,'14.7'), React.createElement('td',null,'23.2'), React.createElement('td',null,'36.5'), React.createElement('td',null,'30.4'), React.createElement('td',null,'39.2'), React.createElement('td',null,'27.8'), React.createElement('td',null,'54.1'), React.createElement('td',null,'47.6')),
          React.createElement('tr', {className:'our-row'}, React.createElement('td',null,'TTARAG'), React.createElement('td',{className:'best'},'16.4'), React.createElement('td',null,'25.8'), React.createElement('td',{className:'best'},'40.7'), React.createElement('td',{className:'best'},'33.8'), React.createElement('td',null,'41.1'), React.createElement('td',{className:'best'},'30.5'), React.createElement('td',{className:'best'},'71.8'), React.createElement('td',{className:'best'},'54.0')),
          React.createElement('tr', {style:{background:'#e8f5e9'}}, React.createElement('td',null,'Δ vs naive'), React.createElement('td',{className:'delta-pos'},'+1.7'), React.createElement('td',{className:'delta-pos'},'+2.6'), React.createElement('td',{className:'delta-pos'},'+4.2'), React.createElement('td',{className:'delta-pos'},'+3.4'), React.createElement('td',{className:'delta-pos'},'+1.9'), React.createElement('td',{className:'delta-pos'},'+2.7'), React.createElement('td',{className:'delta-pos'},'+17.7'), React.createElement('td',{className:'delta-pos'},'+6.4')),
          React.createElement('tr', null, React.createElement('td', {colSpan:9, style:{fontWeight:700, fontStyle:'italic', background:'#f0f0f0'}}, 'ChatGLM-3-6b')),
          React.createElement('tr', null, React.createElement('td',null,'naive-rag'), React.createElement('td',null,'9.8'), React.createElement('td',null,'18.7'), React.createElement('td',null,'31.4'), React.createElement('td',null,'22.4'), React.createElement('td',null,'33.4'), React.createElement('td',null,'22.0'), React.createElement('td',null,'51.4'), React.createElement('td',null,'19.8')),
          React.createElement('tr', {className:'our-row'}, React.createElement('td',null,'TTARAG'), React.createElement('td',{className:'best'},'14.0'), React.createElement('td',{className:'best'},'22.1'), React.createElement('td',{className:'best'},'33.5'), React.createElement('td',null,'25.5'), React.createElement('td',{className:'best'},'38.1'), React.createElement('td',{className:'best'},'25.7'), React.createElement('td',{className:'best'},'58.4'), React.createElement('td',{className:'best'},'44.8')),
          React.createElement('tr', {style:{background:'#e8f5e9'}}, React.createElement('td',null,'Δ vs naive'), React.createElement('td',{className:'delta-pos'},'+4.2'), React.createElement('td',{className:'delta-pos'},'+3.4'), React.createElement('td',{className:'delta-pos'},'+2.1'), React.createElement('td',{className:'delta-pos'},'+3.1'), React.createElement('td',{className:'delta-pos'},'+4.7'), React.createElement('td',{className:'delta-pos'},'+3.7'), React.createElement('td',{className:'delta-pos'},'+7.0'), React.createElement('td',{className:'delta-pos'},'+25.0'))
        )
      )
    ),
  },
  sota_table: {
    title: 'vs. Pretrained RAG Models (Llama-2-7b backbone)',
    color: 'purple',
    grow: false,
    body: React.createElement('table', null,
      React.createElement('thead', null,
        React.createElement('tr', null,
          React.createElement('th', null, 'Model'),
          React.createElement('th', null, 'Finance'),
          React.createElement('th', null, 'Sports'),
          React.createElement('th', null, 'Music'),
          React.createElement('th', null, 'Movie'),
          React.createElement('th', null, 'Open'),
          React.createElement('th', null, 'Overall'),
          React.createElement('th', null, 'BioASQ'),
          React.createElement('th', null, 'PubMed')
        )
      ),
      React.createElement('tbody', null,
        React.createElement('tr', null, React.createElement('td',null,'Naive-RAG'), React.createElement('td',null,'14.7'), React.createElement('td',null,'23.2'), React.createElement('td',null,'36.5'), React.createElement('td',null,'30.4'), React.createElement('td',null,'39.2'), React.createElement('td',null,'27.8'), React.createElement('td',null,'54.1'), React.createElement('td',null,'47.6')),
        React.createElement('tr', null, React.createElement('td',null,'Ret-Robust'), React.createElement('td',null,'14.6'), React.createElement('td',null,'20.6'), React.createElement('td',null,'33.2'), React.createElement('td',null,'32.4'), React.createElement('td',null,'33.5'), React.createElement('td',null,'26.1'), React.createElement('td',null,'24.7'), React.createElement('td',null,'28.4')),
        React.createElement('tr', null, React.createElement('td',null,'RAAT'), React.createElement('td',null,'13.4'), React.createElement('td',null,'18.1'), React.createElement('td',null,'28.6'), React.createElement('td',null,'25.2'), React.createElement('td',null,'31.7'), React.createElement('td',null,'22.7'), React.createElement('td',null,'64.9'), React.createElement('td',null,'46.6')),
        React.createElement('tr', null, React.createElement('td',null,'Self-RAG'), React.createElement('td',null,'11.4'), React.createElement('td',null,'19.8'), React.createElement('td',null,'22.5'), React.createElement('td',null,'20.9'), React.createElement('td',null,'26.7'), React.createElement('td',null,'19.8'), React.createElement('td',null,'57.1'), React.createElement('td',null,'43.4')),
        React.createElement('tr', {className:'our-row'}, React.createElement('td',null,'TTARAG'), React.createElement('td',{className:'best'},'16.4'), React.createElement('td',{className:'best'},'25.8'), React.createElement('td',{className:'best'},'40.7'), React.createElement('td',{className:'best'},'33.8'), React.createElement('td',{className:'best'},'41.1'), React.createElement('td',{className:'best'},'30.5'), React.createElement('td',{className:'best'},'71.8'), React.createElement('td',{className:'best'},'54.0'))
      )
    ),
  },
  hyperparams: {
    title: 'Hyperparameter Analysis',
    color: 'orange',
    grow: true,
    body: React.createElement(React.Fragment, null,
      React.createElement('div', {className:'fig'}, React.createElement('div', {className:'fig-wrap'}, React.createElement('img', {src:'accuracy_vs_lr.png', alt:'Accuracy vs Learning Rate'})),
        React.createElement('div', {className:'cap'}, React.createElement('b', null, 'Figure 2. '), 'Accuracy vs. Learning Rate — optimal at 1e-6 to 1e-5')
      ),
      React.createElement('div', {className:'fig', style:{marginTop:'2mm'}}, React.createElement('div', {className:'fig-wrap'}, React.createElement('img', {src:'accuracy_vs_pairs.png', alt:'Accuracy vs Pairs'})),
        React.createElement('div', {className:'cap'}, React.createElement('b', null, 'Figure 3. '), 'Accuracy vs. Number of Adaptation Pairs — 3-5 pairs optimal')
      )
    ),
  },
  ablation: {
    title: 'Ablation & Efficiency',
    color: 'orange',
    grow: false,
    body: React.createElement(React.Fragment, null,
      React.createElement('div', {className:'hl hl-orange'}, React.createElement('p', null, 'Segmentation strategy yields consistent gains: +1.1% (Llama-3.1), +0.4% (Llama-2), +0.7% (ChatGLM)')),
      React.createElement('table', null,
        React.createElement('thead', null, React.createElement('tr', null,
          React.createElement('th', null, 'Metric'), React.createElement('th', null, '1 pair'), React.createElement('th', null, '3 pairs'), React.createElement('th', null, '5 pairs'), React.createElement('th', null, 'CoT'), React.createElement('th', null, 'Naive')
        )),
        React.createElement('tbody', null,
          React.createElement('tr', null, React.createElement('td',null,'Total (s)'), React.createElement('td',null,'4,740'), React.createElement('td',null,'6,621'), React.createElement('td',null,'7,023'), React.createElement('td',null,'11,688'), React.createElement('td',null,'961')),
          React.createElement('tr', null, React.createElement('td',null,'Avg (s)'), React.createElement('td',null,'1.75'), React.createElement('td',null,'2.45'), React.createElement('td',null,'2.60'), React.createElement('td',null,'4.32'), React.createElement('td',null,'0.36'))
        )
      ),
      React.createElement('ul', null,
        React.createElement('li', null, 'TTARAG is ', React.createElement('b', null, '40% faster'), ' than CoT while achieving better performance'),
        React.createElement('li', null, 'Computational overhead remains practical for real-world deployment')
      )
    ),
  },
  concl: {
    title: 'Conclusion & Links',
    color: 'red',
    grow: false,
    body: React.createElement(React.Fragment, null,
      React.createElement('ul', null,
        React.createElement('li', null, 'TTARAG enables ', React.createElement('b', null, 'fully unsupervised domain adaptation'), ' for RAG at inference time'),
        React.createElement('li', null, 'Simple prefix-suffix prediction objective provides effective self-supervised learning signals'),
        React.createElement('li', null, 'Achieves up to ', React.createElement('b', null, '+25%'), ' improvement on specialized domains (medical, finance)'),
        React.createElement('li', null, 'Computationally efficient — significantly faster than Chain-of-Thought')
      ),
      React.createElement('div', {className:'links'},
        React.createElement('div', {className:'ll'}, React.createElement('b', null, 'Code: '), 'github.com/sunxin000/TTARAG', React.createElement('br'), React.createElement('b', null, 'Paper: '), 'arXiv:2601.11443')
      )
    ),
  },
};
