const stages = ['conversations', 'summaries', 'clusters', 'meta_clusters', 'dimensionality'];
const state = { view: 'overview', data: Object.fromEntries(stages.map(key => [key, []])), source: 'No dataset loaded', search: '', level: 'all' };
const $ = selector => document.querySelector(selector);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const count = value => Number(value || 0).toLocaleString();
const allClusters = () => state.data.dimensionality.length ? state.data.dimensionality : [...state.data.clusters, ...state.data.meta_clusters];
const baseClusters = () => allClusters().filter(item => (item.level ?? 0) === 0);
const hasData = () => state.data.conversations.length || state.data.summaries.length || allClusters().length;
const summaryById = () => new Map(state.data.summaries.map(item => [item.chat_id, item]));
const conversationById = () => new Map(state.data.conversations.map(item => [item.chat_id, item]));
const clusterCount = item => item.chat_ids?.length ?? item.count ?? 0;
const button = (label, handler = 'import') => `<button class="button button-dark" data-action="${handler}">${label} <span>↗</span></button>`;
function empty(title, description) { return `<div class="empty"><div class="empty-mark">◇</div><h2>${title}</h2><p>${description}</p>${button('Import analysis')}</div>`; }
function pageIntro(kicker, title, subtitle) { return `<div class="page-intro"><div><span class="eyebrow">${kicker}</span><h1>${title}</h1><p>${subtitle}</p></div></div>`; }
function card(item) {
  const level = item.level || 0;
  return `<button class="pattern-card" data-cluster="${escapeHTML(item.id)}"><div class="card-top"><span class="pill">${level ? `LEVEL ${level + 1} THEME` : 'BASE PATTERN'}</span><span class="count">${count(clusterCount(item))} chats</span></div><h3>${escapeHTML(item.name)}</h3><p>${escapeHTML(item.description)}</p><span class="arrow">↗</span></button>`;
}
function renderOverview() {
  if (!hasData()) return `<div class="hero"><div><span class="eyebrow">CONVERSATION INTELLIGENCE</span><h1>See the patterns behind every conversation.</h1><p>Analyze chat data, group similar needs, and explore the themes that matter most.</p></div>${button('Import an analysis')}</div><div style="margin-top:20px">${empty('Your workspace is ready', 'Run the CLI on a conversation file or import saved analysis files to explore patterns here.')}</div>`;
  const clusters = baseClusters().sort((a,b) => clusterCount(b)-clusterCount(a));
  const largest = clusters[0];
  const total = state.data.conversations.length || new Set(clusters.flatMap(item => item.chat_ids || [])).size;
  const grouped = new Set(clusters.flatMap(item => item.chat_ids || [])).size;
  const depth = Math.max(0, ...allClusters().map(item => item.level || 0));
  const bars = clusters.slice(0, 5).map(item => `<button class="bar-row" data-cluster="${escapeHTML(item.id)}"><span class="truncate">${escapeHTML(item.name)}</span><span class="bar-track" aria-hidden="true"><span class="bar-fill" style="width:${Math.max(4,100*clusterCount(item)/(clusterCount(largest)||1))}%"></span></span><b>${count(clusterCount(item))}</b></button>`).join('');
  return `<div class="hero"><div><span class="eyebrow">ANALYSIS OVERVIEW</span><h1>Explore what people ask for.</h1><p>See which requests recur, inspect individual chats, and move between levels of the theme map.</p></div>${button('Explore patterns', 'patterns')}</div>
  <div class="metrics"><div class="metric"><small>Conversations</small><strong>${count(total)}</strong></div><div class="metric"><small>Base patterns</small><strong>${count(clusters.length)}</strong></div><div class="metric"><small>Grouped coverage</small><strong>${total ? Math.round(100*grouped/total) : 0}%</strong></div><div class="metric"><small>Theme levels</small><strong>${depth+1}</strong></div></div>
  <section class="panel group-panel"><h3>Largest conversation groups</h3><p class="group-caption">Each bar compares the number of conversations in a base pattern. Select a group to read its details.</p>${bars || '<p class="muted">No groups yet.</p>'}</section>
  <div class="section-head"><h2>Patterns worth exploring</h2><button data-action="patterns">View all ↗</button></div><div class="card-grid">${clusters.slice(0,6).map(card).join('')}</div>`;
}
function renderPatterns() {
  const all = allClusters().sort((a,b) => (b.level||0)-(a.level||0) || clusterCount(b)-clusterCount(a));
  const filtered = all.filter(item => (state.level === 'all' || String(item.level||0) === state.level) && `${item.name} ${item.description}`.toLowerCase().includes(state.search.toLowerCase()));
  const levels = [...new Set(all.map(item => item.level || 0))].sort((a,b)=>a-b);
  return `${pageIntro('PATTERN LIBRARY', 'Recurring needs, clearly grouped.', 'Search across themes and open any pattern to see the conversations behind it.')}${!hasData() ? empty('No patterns yet', 'Import an analysis to browse its themes.') : `<div class="toolbar"><input id="pattern-search" class="search" type="search" placeholder="Search patterns..." value="${escapeHTML(state.search)}" aria-label="Search patterns"/><select id="level-filter" class="select" aria-label="Filter by level"><option value="all">All levels</option>${levels.map(level => `<option value="${level}" ${state.level===String(level)?'selected':''}>${level===0?'Base patterns':`Level ${level+1} themes`}</option>`).join('')}</select><span class="muted">${filtered.length} results</span></div>${filtered.length ? `<div class="card-grid">${filtered.map(card).join('')}</div>` : empty('No matching patterns', 'Try a different search or level.')}`}`;
}
function mapPosition(value, min, max, padding) { return min === max ? 50 : padding + (100 - 2 * padding) * (value-min)/(max-min); }
function renderMap() {
  const all = allClusters();
  const levels = [...new Set(all.map(item => item.level || 0))].sort((a,b)=>a-b);
  const level = state.level === 'all' ? (levels.includes(0) ? 0 : levels[0]) : Number(state.level);
  const items = all.filter(item => (item.level||0)===level);
  if (!hasData()) return `${pageIntro('THEME MAP','A spatial view of your themes.','Nearby points represent related cluster descriptions.')}${empty('No map yet','Import an analysis with projected clusters to view the map.')}`;
  const xs = items.map(item => Number(item.x_coord || 0)); const ys = items.map(item => Number(item.y_coord || 0));
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const padding = items.length <= 3 ? 28 : 13;
  const points = items.map((item,index) => { const x=mapPosition(xs[index],minX,maxX,padding), y=100-mapPosition(ys[index],minY,maxY,padding); const radius=Math.min(21,12+Math.sqrt(clusterCount(item)||1)*2); return `<g data-cluster="${escapeHTML(item.id)}" tabindex="0" role="button" aria-label="${escapeHTML(item.name)}"><circle class="point" cx="${x}%" cy="${y}%" r="${radius}"/><text class="point-label" x="${x}%" y="${y}%" dx="${radius+5}" dy="3">${escapeHTML(item.name.slice(0,24))}</text></g>`; }).join('');
  return `${pageIntro('THEME MAP','See how themes relate.','Each point is a conversation pattern. Select a level, then open a point for details.')}<div class="toolbar"><select id="map-level" class="select" aria-label="Map level">${levels.map(item=>`<option value="${item}" ${item===level?'selected':''}>${item===0?'Base patterns':`Level ${item+1} themes`}</option>`).join('')}</select><span class="muted">${items.length} patterns at this level</span></div><div class="map-wrap"><div class="map-panel"><svg viewBox="0 0 900 490" preserveAspectRatio="none" role="img" aria-label="Cluster map"><line class="axis" x1="50%" y1="0" x2="50%" y2="100%"/><line class="axis" x1="0" y1="50%" x2="100%" y2="50%"/>${points}</svg></div><div class="map-side"><h3>Patterns in view</h3>${items.sort((a,b)=>clusterCount(b)-clusterCount(a)).map(item=>`<button data-cluster="${escapeHTML(item.id)}"><span class="map-dot"></span><span>${escapeHTML(item.name)}<br><small class="muted">${count(clusterCount(item))} conversations</small></span></button>`).join('')}</div></div>`;
}
function renderConversations() {
  const summaries=summaryById(); const conversations=state.data.conversations;
  const filtered=conversations.filter(item => `${item.chat_id} ${summaries.get(item.chat_id)?.summary||''} ${item.messages?.map(message=>message.content).join(' ')||''}`.toLowerCase().includes(state.search.toLowerCase()));
  return `${pageIntro('CONVERSATION EXPLORER','Go back to the source.','Review summaries and the underlying messages for each conversation.')}${!hasData()?empty('No conversations yet','Import an analysis with conversation data to browse messages.'):`<div class="toolbar"><input id="conversation-search" class="search" type="search" placeholder="Search conversations..." value="${escapeHTML(state.search)}" aria-label="Search conversations"/><span class="muted">${filtered.length} results</span></div>${filtered.length?`<div class="conversation-list">${filtered.map((item,index)=>`<button class="conversation-row" data-conversation="${escapeHTML(item.chat_id)}"><span class="row-num">${String(index+1).padStart(2,'0')}</span><span class="row-main"><strong>${escapeHTML(summaries.get(item.chat_id)?.summary||item.messages?.find(message=>message.role==='user')?.content||item.chat_id)}</strong><small>${escapeHTML(item.chat_id)} · ${item.messages?.length||0} messages</small></span><span class="row-arrow">↗</span></button>`).join('')}</div>`:empty('No matching conversations','Try a different search term.')}`}`;
}
function render() {
  $('.nav-item.active')?.classList.remove('active'); $(`.nav-item[data-view="${state.view}"]`)?.classList.add('active');
  $('#view-name').textContent=state.view[0].toUpperCase()+state.view.slice(1);
  $('#record-count').textContent=`${count(state.data.conversations.length)} conversations`;
  $('#dataset-label').textContent=state.source;
  $('#app-content').innerHTML=({overview:renderOverview,patterns:renderPatterns,map:renderMap,conversations:renderConversations})[state.view]();
}
function openCluster(id) {
  const item=allClusters().find(cluster=>cluster.id===id); if(!item)return;
  const summaries=summaryById(), conversations=conversationById();
  $('#detail-kicker').textContent=`LEVEL ${(item.level||0)+1} PATTERN`;
  $('#detail-title').textContent=item.name;
  const parent=allClusters().find(cluster=>cluster.id===item.parent_id);
  $('#detail-body').innerHTML=`<p class="detail-description">${escapeHTML(item.description)}</p><div class="detail-meta"><span>${count(clusterCount(item))} conversations</span><span>${parent?`Part of ${escapeHTML(parent.name)}`:'Top-level theme'}</span></div><div class="detail-section"><h3>Conversations in this pattern</h3>${(item.chat_ids||[]).slice(0,100).map(chatId=>{const summary=summaries.get(chatId);const conversation=conversations.get(chatId);return `<div class="detail-item"><strong>${escapeHTML(summary?.summary||chatId)}</strong><p>${escapeHTML(summary?.request||'')}</p>${conversation?`<button class="button button-quiet" data-conversation="${escapeHTML(chatId)}" style="margin-top:9px">Read conversation ↗</button>`:''}</div>`}).join('')||'<p class="muted">No linked conversations.</p>'}</div>`;
  $('#detail-dialog').showModal();
}
function openConversation(id) {
  const item=conversationById().get(id); if(!item)return;
  const summary=summaryById().get(id);
  $('#detail-kicker').textContent='CONVERSATION'; $('#detail-title').textContent=summary?.summary||id;
  $('#detail-body').innerHTML=`<p class="detail-description">${escapeHTML(summary?.request||'Original messages')}</p><div class="detail-meta"><span>${escapeHTML(id)}</span><span>${item.messages?.length||0} messages</span></div><div class="detail-section"><h3>Transcript</h3>${(item.messages||[]).map(message=>`<div class="message ${message.role==='user'?'user':'assistant'}"><b>${escapeHTML(message.role)}</b>${escapeHTML(message.content)}</div>`).join('')}</div>`;
  $('#detail-dialog').showModal();
}
function previewData() {
  const names=['Troubleshoot login issues','Find billing information','Request account exports','Understand API limits','Configure team permissions','Improve search results'];
  const conversations=Array.from({length:24},(_,i)=>({chat_id:`sample-${i+1}`,messages:[{role:'user',content:`Can you help me with ${names[i%names.length].toLowerCase()}?`},{role:'assistant',content:`Here is a way to address ${names[i%names.length].toLowerCase()}.`}],metadata:{channel:i%2?'web':'support'}}));
  const summaries=conversations.map((item,i)=>({chat_id:item.chat_id,summary:`The user asked for help with ${names[i%names.length].toLowerCase()}.`,request:names[i%names.length],metadata:item.metadata}));
  const clusters=names.map((name,i)=>({id:`sample-cluster-${i}`,name,description:`Conversations about ${name.toLowerCase()} and the steps users need to take.`,slug:name.toLowerCase().replaceAll(' ','_'),chat_ids:conversations.filter((_,j)=>j%names.length===i).map(item=>item.chat_id),parent_id:`sample-parent-${Math.floor(i/3)}`,level:0,x_coord:Math.cos(i*1.05)*10,y_coord:Math.sin(i*1.05)*10}));
  const meta_clusters=[{id:'sample-parent-0',name:'Resolve access and account questions',description:'Requests about signing in, payments, and data access.',chat_ids:clusters.slice(0,3).flatMap(item=>item.chat_ids),parent_id:null,level:1,x_coord:-5,y_coord:1},{id:'sample-parent-1',name:'Improve product setup and discovery',description:'Requests about API usage, permissions, and search.',chat_ids:clusters.slice(3).flatMap(item=>item.chat_ids),parent_id:null,level:1,x_coord:5,y_coord:1}];
  state.data={conversations,summaries,clusters,meta_clusters,dimensionality:[...clusters,...meta_clusters]}; state.source='Preview dataset'; state.view='overview'; state.search=''; state.level='all'; render();
}
async function importFiles(files) {
  const next={...state.data};
  try {
    for(const file of files) {
      const raw=(await file.text()).trim(); if(!raw)continue;
      let value;
      if(raw.startsWith('{') && raw.split('\n').length>1 && file.name.endsWith('.jsonl')) value=raw.split('\n').filter(Boolean).map(line=>JSON.parse(line));
      else if(raw.startsWith('[')||raw.startsWith('{')) { try {value=JSON.parse(raw)} catch {value=raw.split('\n').filter(Boolean).map(line=>JSON.parse(line))} }
      else value=raw.split('\n').filter(Boolean).map(line=>JSON.parse(line));
      if(value && !Array.isArray(value) && stages.some(key=>key in value)) for(const key of stages) next[key]=Array.isArray(value[key])?value[key]:next[key];
      else {const key=file.name.replace(/\.(jsonl|json)$/,''); if(!stages.includes(key))throw new Error(`Unrecognized file: ${file.name}`); if(!Array.isArray(value))throw new Error(`${file.name} must contain a list`); next[key]=value;}
    }
    state.data=next; state.source=files.length===1?files[0].name:`${files.length} imported files`; state.view='overview'; state.search=''; state.level='all'; $('#import-dialog').close(); render();
  } catch(error) { $('#import-status').textContent=error.message; $('#import-status').classList.add('error'); }
}
document.addEventListener('click',event=>{
  const nav=event.target.closest('[data-view]'); if(nav){state.view=nav.dataset.view;state.search='';state.level='all';render();return;}
  const action=event.target.closest('[data-action]'); if(action){if(action.dataset.action==='import')$('#import-dialog').showModal();else{state.view=action.dataset.action;state.search='';state.level='all';render();}return;}
  const cluster=event.target.closest('[data-cluster]'); if(cluster){openCluster(cluster.dataset.cluster);return;}
  const conversation=event.target.closest('[data-conversation]'); if(conversation){$('#detail-dialog').close();openConversation(conversation.dataset.conversation);}
});
document.addEventListener('input',event=>{if(event.target.id==='pattern-search'||event.target.id==='conversation-search'){state.search=event.target.value;const position=event.target.selectionStart;render();const selector=event.target.id==='pattern-search'?'#pattern-search':'#conversation-search';$(selector)?.focus();$(selector)?.setSelectionRange(position,position);}});
document.addEventListener('change',event=>{if(event.target.id==='level-filter'||event.target.id==='map-level'){state.level=event.target.value;render();}});
$('#import-top').onclick=$('#import-sidebar').onclick=()=>$('#import-dialog').showModal();
$('#sample-button').onclick=()=>{$('#import-dialog').close();previewData()};
$('#file-input').onchange=event=>importFiles([...event.target.files]);
const dropzone=$('#dropzone');
dropzone.addEventListener('dragover',event=>{event.preventDefault();dropzone.classList.add('drag')});
dropzone.addEventListener('dragleave',()=>dropzone.classList.remove('drag'));
dropzone.addEventListener('drop',event=>{event.preventDefault();dropzone.classList.remove('drag');importFiles([...event.dataTransfer.files])});
fetch('/api/analysis').then(response=>response.ok?response.json():null).then(data=>{if(data&&stages.some(key=>data[key]?.length)){state.data=Object.fromEntries(stages.map(key=>[key,data[key]||[]]));state.source='Local checkpoints';render()}}).catch(()=>{});
render();
