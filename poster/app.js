const { useState, useRef, useEffect, useCallback } = React;
const MM = 3.7795275591;
const POSTER_W_MM = 841;
const POSTER_H_MM = 1189;

/* A0 portrait: 3 columns layout */
const DEFAULT_LAYOUT = {
  columns: [
    { id: 'col1', widthMm: 265, cards: ['tldr', 'motivation', 'framework'] },
    { id: 'col2', widthMm: null, cards: ['method', 'results_table', 'sota_table'] },
    { id: 'col3', widthMm: 265, cards: ['hyperparams', 'ablation', 'concl'] },
  ],
};
const DEFAULT_CARD_HEIGHTS = {};
const DEFAULT_FONT_SCALE = 1.5;
const DEFAULT_LOGOS = [];

function cloneLayout(layout) {
  return { columns: layout.columns.map(function(c) { return { id: c.id, widthMm: c.widthMm, cards: c.cards.slice() }; }) };
}

var LS_KEY = 'posterConfig';
function getFullConfig(layout, cardHeights, fontScale, logos) {
  return { columns: layout.columns, cardHeights: cardHeights, fontScale: fontScale, logos: logos };
}
function loadInitialConfig() {
  try { var s = localStorage.getItem(LS_KEY); if (s) return JSON.parse(s); } catch(e) {}
  return null;
}
var INIT_CONFIG = loadInitialConfig();

function PosterApp() {
  var _s = useState(INIT_CONFIG ? cloneLayout({columns: INIT_CONFIG.columns}) : cloneLayout(DEFAULT_LAYOUT));
  var layout = _s[0], setLayout = _s[1];
  var _s2 = useState(null);
  var selectedCard = _s2[0], setSelectedCard = _s2[1];
  var _s3 = useState(INIT_CONFIG ? INIT_CONFIG.cardHeights || {} : Object.assign({}, DEFAULT_CARD_HEIGHTS));
  var cardHeights = _s3[0], setCardHeights = _s3[1];
  var _s4 = useState(false);
  var preview = _s4[0], setPreview = _s4[1];
  var _s5 = useState(INIT_CONFIG ? INIT_CONFIG.logos || DEFAULT_LOGOS : DEFAULT_LOGOS);
  var logos = _s5[0], setLogos = _s5[1];
  var _s6 = useState(INIT_CONFIG ? INIT_CONFIG.fontScale || DEFAULT_FONT_SCALE : DEFAULT_FONT_SCALE);
  var fontScale = _s6[0], setFontScaleState = _s6[1];
  var currentScaleRef = useRef(1);
  var posterRef = useRef(null);

  useEffect(function() { var cfg = getFullConfig(layout, cardHeights, fontScale, logos); localStorage.setItem(LS_KEY, JSON.stringify(cfg)); }, [layout, cardHeights, fontScale, logos]);
  useEffect(function() { document.documentElement.style.setProperty('--font-scale', fontScale); }, [fontScale]);
  useEffect(function() { document.body.classList.toggle('preview', preview); }, [preview]);

  var fit = useCallback(function() {
    if (window.matchMedia('print').matches) return;
    var pW = POSTER_W_MM * MM, pH = POSTER_H_MM * MM;
    var scale = Math.min(window.innerWidth / pW, window.innerHeight / pH);
    currentScaleRef.current = scale;
    var left = (window.innerWidth - pW * scale) / 2;
    var top = (window.innerHeight - pH * scale) / 2;
    document.body.style.transform = 'translate(' + left + 'px,' + top + 'px) scale(' + scale + ')';
  }, []);
  useEffect(function() { fit(); window.addEventListener('resize', fit); return function() { window.removeEventListener('resize', fit); }; }, [fit]);

  useEffect(function() { if (typeof renderMathInElement === 'function') renderMathInElement(document.body, {delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]}); });

  useEffect(function() {
    function handler(e) { if (selectedCard && !e.target.closest('.swap-handle') && !e.target.closest('.drop-zone')) setSelectedCard(null); }
    document.addEventListener('click', handler); return function() { document.removeEventListener('click', handler); };
  }, [selectedCard]);

  function swapCards(id1, id2) {
    if (id1 === id2) return false;
    setLayout(function(prev) { var next = cloneLayout(prev); var l1, l2; for (var ci=0;ci<next.columns.length;ci++) { var c=next.columns[ci]; var i1=c.cards.indexOf(id1); if(i1!==-1)l1={col:c,idx:i1}; var i2=c.cards.indexOf(id2); if(i2!==-1)l2={col:c,idx:i2}; } if(!l1||!l2) return prev; l1.col.cards[l1.idx]=id2; l2.col.cards[l2.idx]=id1; return next; });
    setCardHeights(function(prev) { var n=Object.assign({},prev); delete n[id1]; delete n[id2]; return n; });
    return true;
  }
  function moveCard(cardId, targetColId, position) {
    setLayout(function(prev) { var next = cloneLayout(prev); for (var ci=0;ci<next.columns.length;ci++) { var i=next.columns[ci].cards.indexOf(cardId); if(i!==-1){next.columns[ci].cards.splice(i,1);break;} } var tc=next.columns.find(function(c){return c.id===targetColId;}); if(!tc)return prev; tc.cards.splice(Math.max(0,Math.min(position,tc.cards.length)),0,cardId); return next; });
    setCardHeights(function(prev) { var n=Object.assign({},prev); delete n[cardId]; return n; });
  }
  function setColumnWidth(colId, widthMm) { setLayout(function(prev) { var next=cloneLayout(prev); var c=next.columns.find(function(c){return c.id===colId;}); if(!c) return prev; c.widthMm=widthMm; return next; }); }
  function setCardHeight(cardId, heightMm) { setCardHeights(function(prev) { return Object.assign({},prev,{[cardId]:heightMm}); }); }
  function getWaste() { var total=0; var details=[]; document.querySelectorAll('.fig-wrap').forEach(function(fw) { var img=fw.querySelector('img'); if(!img)return; var fr=fw.getBoundingClientRect(),ir=img.getBoundingClientRect(); var wH=Math.abs(fr.height-ir.height),wW=Math.abs(fr.width-ir.width); total+=wH+wW; var card=fw.closest('.card'); details.push({card:card?card.dataset.id:'',wasteH:Math.round(wH),wasteW:Math.round(wW),pct:Math.round((wH+wW)/(fr.height+fr.width)*100)}); }); return {total:Math.round(total),details:details}; }
  function getLayout() { var s=currentScaleRef.current; var r=[]; document.querySelectorAll('#poster > .col').forEach(function(col) { var cards=[]; col.querySelectorAll('.card').forEach(function(c) { var b=c.getBoundingClientRect(); cards.push({id:c.dataset.id,height:Math.round(b.height/s),heightMm:Math.round(b.height/s/MM),grow:c.classList.contains('grow')}); }); r.push({colId:col.id,widthPx:Math.round(col.getBoundingClientRect().width/s),widthMm:Math.round(col.getBoundingClientRect().width/s/MM),cards:cards}); }); return r; }
  function resetLayout() { setLayout(cloneLayout(DEFAULT_LAYOUT)); setCardHeights(Object.assign({},DEFAULT_CARD_HEIGHTS)); setFontScaleState(DEFAULT_FONT_SCALE); setLogos(DEFAULT_LOGOS.slice()); setSelectedCard(null); localStorage.removeItem(LS_KEY); }
  function saveConfig() { var cfg=getFullConfig(layout,cardHeights,fontScale,logos); var b=new Blob([JSON.stringify(cfg,null,2)+'\n'],{type:'application/json'}); var u=URL.createObjectURL(b); var a=document.createElement('a'); a.href=u; a.download='poster-config.json'; a.click(); URL.revokeObjectURL(u); }
  function copyConfig() { navigator.clipboard.writeText(JSON.stringify(getFullConfig(layout,cardHeights,fontScale,logos),null,2)).then(function(){alert('Config copied!');}); }
  function setFontScale(s) { setFontScaleState(parseFloat(s)||1.5); }

  useEffect(function() { window.posterAPI = { swapCards:swapCards, moveCard:moveCard, setColumnWidth:setColumnWidth, setCardHeight:setCardHeight, setFontScale:setFontScale, getWaste:getWaste, getLayout:getLayout, getConfig:function(){return getFullConfig(layout,cardHeights,fontScale,logos);}, resetLayout:resetLayout, saveConfig:saveConfig, copyConfig:copyConfig }; });

  function handleSwapClick(cardId, e) { e.stopPropagation(); if (!selectedCard) setSelectedCard(cardId); else if (selectedCard===cardId) setSelectedCard(null); else { swapCards(selectedCard, cardId); setSelectedCard(null); } }
  function handleDropZone(targetColId, position, e) { e.stopPropagation(); if (!selectedCard) return; moveCard(selectedCard, targetColId, position); setSelectedCard(null); }

  function handleColResize(dividerIdx, e) {
    e.preventDefault(); var handle=e.currentTarget; handle.classList.add('active');
    var targetColIdx=dividerIdx===0?0:2; var invert=dividerIdx===0?1:-1;
    var targetEl=document.getElementById(layout.columns[targetColIdx].id);
    var startX=e.clientX; var scale=currentScaleRef.current;
    var startW=targetEl.getBoundingClientRect().width/scale;
    function onMove(ev) { var dx=(ev.clientX-startX)/scale*invert; var newW=Math.max(120,(startW+dx)/MM); setLayout(function(prev) { var next=cloneLayout(prev); next.columns[targetColIdx].widthMm=Math.round(newW); return next; }); }
    function onUp() { document.removeEventListener('mousemove',onMove); document.removeEventListener('mouseup',onUp); handle.classList.remove('active'); document.body.style.cursor=''; }
    document.body.style.cursor='col-resize'; document.addEventListener('mousemove',onMove); document.addEventListener('mouseup',onUp);
  }

  function handleRowResize(colId, cardAboveIdx, e) {
    e.preventDefault(); var handle=e.currentTarget; handle.classList.add('active');
    var col=layout.columns.find(function(c){return c.id===colId;}); var aboveId=col.cards[cardAboveIdx];
    var aboveEl=document.querySelector('[data-id="'+aboveId+'"]'); if(!aboveEl)return;
    var startY=e.clientY; var scale=currentScaleRef.current; var startH=aboveEl.getBoundingClientRect().height/scale;
    setCardHeights(function(prev){return Object.assign({},prev,{[aboveId]:startH/MM});});
    function onMove(ev) { var dy=(ev.clientY-startY)/scale; setCardHeights(function(prev){return Object.assign({},prev,{[aboveId]:Math.max(20,(startH+dy)/MM)});}); }
    function onUp() { document.removeEventListener('mousemove',onMove); document.removeEventListener('mouseup',onUp); handle.classList.remove('active'); document.body.style.cursor=''; }
    document.body.style.cursor='row-resize'; document.addEventListener('mousemove',onMove); document.addEventListener('mouseup',onUp);
  }

  function isCardGrow(cardId) {
    if (cardHeights[cardId] != null) return false;
    for (var ci=0;ci<layout.columns.length;ci++) {
      var col=layout.columns[ci]; var idx=col.cards.indexOf(cardId); if(idx===-1) continue;
      var hasGrower=col.cards.some(function(cid){return CARD_REGISTRY[cid] && CARD_REGISTRY[cid].grow && cardHeights[cid]==null;});
      if(hasGrower) return (CARD_REGISTRY[cardId] && CARD_REGISTRY[cardId].grow) || false;
      var lastFlexible=null; for(var i=col.cards.length-1;i>=0;i--){ if(cardHeights[col.cards[i]]==null){lastFlexible=col.cards[i];break;} }
      return cardId === lastFlexible;
    }
    return false;
  }

  function renderCard(cardId) {
    var card = CARD_REGISTRY[cardId]; if (!card) return null;
    var classes = ['card']; if (card.color) classes.push(card.color); if (isCardGrow(cardId)) classes.push('grow'); if (selectedCard===cardId) classes.push('swap-selected');
    var style = {}; if (cardHeights[cardId]!=null) style.flex='0 0 '+cardHeights[cardId]+'mm';
    return React.createElement('div', {key:cardId, className:classes.join(' '), 'data-id':cardId, style:style},
      React.createElement('div', {className:'swap-handle', onClick:function(e){handleSwapClick(cardId,e);}}, '\u2725'),
      React.createElement('h2', null, card.title),
      card.body
    );
  }

  function renderDropZone(colId, position) {
    var visible = selectedCard !== null;
    return React.createElement('div', {key:'dz-'+colId+'-'+position, className:'drop-zone'+(visible?' visible':''), onClick:function(e){handleDropZone(colId,position,e);}},
      visible && React.createElement('div', {className:'drop-zone-inner'})
    );
  }

  function renderColumn(col) {
    var style = col.widthMm != null ? {flex:'0 0 '+col.widthMm+'mm'} : {flex:'1.5'};
    var children = [renderDropZone(col.id, 0)];
    col.cards.forEach(function(cardId, i) {
      children.push(renderCard(cardId));
      if (i < col.cards.length - 1) children.push(React.createElement('div', {key:'row-'+col.id+'-'+i, className:'divider row-resize', onMouseDown:function(e){handleRowResize(col.id,i,e);}}));
      children.push(renderDropZone(col.id, i + 1));
    });
    return React.createElement('div', {key:col.id, className:'col', id:col.id, style:style}, children);
  }

  return React.createElement(React.Fragment, null,
    React.createElement('div', {className:'toolbar'},
      React.createElement('button', {onClick:function(){setPreview(!preview);}, style:preview?{background:'#1a6fb5',color:'#fff'}:{}}, preview?'Edit':'Preview'),
      !preview && React.createElement(React.Fragment, null,
        React.createElement('button', {onClick:function(){setFontScale(Math.max(0.8,fontScale-0.1));}}, 'A-'),
        React.createElement('button', {onClick:function(){setFontScale(fontScale+0.1);}}, 'A+'),
        React.createElement('button', {onClick:saveConfig, title:'Download poster-config.json'}, 'Save'),
        React.createElement('button', {onClick:copyConfig, title:'Copy config JSON to clipboard'}, 'Copy Config'),
        React.createElement('button', {onClick:resetLayout}, 'Reset')
      )
    ),
    /* Header */
    React.createElement('div', {className:'header'},
      React.createElement('div', {className:'header-left'},
        React.createElement('h1', null, React.createElement('span', {className:'m'}, 'TTARAG: '), 'Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation'),
        React.createElement('div', {className:'authors'},
          'Xin Sun', React.createElement('sup', null, '1'), ', ',
          'Zhongqi Chen', React.createElement('sup', null, '2'), ', ',
          'Qiang Liu', React.createElement('sup', null, '1'), ', ',
          'Shu Wu', React.createElement('sup', null, '1'), ', ',
          'Bowen Song', React.createElement('sup', null, '2'), ', ',
          'Weiqiang Wang', React.createElement('sup', null, '2'), ', ',
          'Zilei Wang', React.createElement('sup', null, '3'), ', ',
          'Liang Wang', React.createElement('sup', null, '1')
        ),
        React.createElement('div', {className:'aff'},
          React.createElement('sup', null, '1'), ' NLPR, MAIS, CASIA    ',
          React.createElement('sup', null, '2'), ' Ant Group    ',
          React.createElement('sup', null, '3'), ' USTC'
        )
      ),
      React.createElement('div', {className:'header-right'},
        React.createElement('div', {className:'badge'}, 'ICASSP 2026'),
        React.createElement('div', {className:'qr'}, React.createElement('img', {src:'qr.png', alt:'QR'}), React.createElement('div', {className:'ql'}, 'Code')),
        React.createElement('div', {className:'qr'}, React.createElement('img', {src:'qr-posterskill.png', alt:'Posterskill'}), React.createElement('div', {className:'ql'}, 'Made with posterskill'))
      )
    ),
    /* Poster columns */
    React.createElement('div', {className:'poster', id:'poster', ref:posterRef},
      layout.columns.map(function(col, colIdx) {
        var elements = [];
        if (colIdx > 0) elements.push(React.createElement('div', {key:'col-div-'+colIdx, className:'divider col-resize', onMouseDown:function(e){handleColResize(colIdx-1,e);}}));
        elements.push(renderColumn(col));
        return elements;
      })
    )
  );
}

var root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(PosterApp));
