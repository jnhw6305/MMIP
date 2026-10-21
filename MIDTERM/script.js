const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => entry.isIntersecting && entry.target.classList.add('visible'));
}, { threshold: 0.12 });
$$('.reveal').forEach((el) => observer.observe(el));

const monthLabels = ['最近月', '前 2 月', '前 3 月', '前 4 月', '前 5 月', '前 6 月'];
const tbody = $('#monthRows');
monthLabels.forEach((label) => {
  const row = document.createElement('tr');
  row.innerHTML = `<th>${label}</th>
    <td><input class="delay-input" type="number" min="-2" max="8" value="0" aria-label="${label}延遲狀態" required></td>
    <td><input class="bill-input" type="number" min="-100000" max="2000000" step="1000" value="40000" aria-label="${label}帳單金額" required></td>
    <td><input class="paid-input" type="number" min="0" max="2000000" step="100" value="5000" aria-label="${label}實際付款" required></td>`;
  tbody.appendChild(row);
});

let model;
fetch('model.json').then((r) => {
  if (!r.ok) throw new Error('模型檔案載入失敗');
  return r.json();
}).then((data) => { model = data; }).catch(() => {
  $('#formError').textContent = '模型載入失敗。請使用本機伺服器開啟網站，不要直接雙擊 HTML。';
});

const samples = {
  safe: { limit: 300000, age: 32, delay: [-1,-1,-1,-1,-1,-1], bill: [62000,59000,57000,55000,52000,50000], paid: [12000,11000,10000,10000,9000,9000] },
  risk: { limit: 80000, age: 43, delay: [2,2,1,1,1,0], bill: [79000,76000,72000,68000,64000,60000], paid: [1000,1000,1500,1500,2000,2000] }
};

function fillSample(sample) {
  $('#limit').value = sample.limit;
  $('#age').value = sample.age;
  $$('.delay-input').forEach((el, i) => el.value = sample.delay[i]);
  $$('.bill-input').forEach((el, i) => el.value = sample.bill[i]);
  $$('.paid-input').forEach((el, i) => el.value = sample.paid[i]);
  $('#formError').textContent = '';
}
$('#safeSample').addEventListener('click', () => fillSample(samples.safe));
$('#riskSample').addEventListener('click', () => fillSample(samples.risk));

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));
const featureLabels = {
  LIMIT_BAL:'信用額度', AGE:'年齡', PAY_0:'最近月延遲', PAY_2:'前 2 月延遲', PAY_3:'前 3 月延遲', PAY_4:'前 4 月延遲', PAY_5:'前 5 月延遲', PAY_6:'前 6 月延遲',
  BILL_AMT1:'最近月帳單', BILL_AMT2:'前 2 月帳單', BILL_AMT3:'前 3 月帳單', BILL_AMT4:'前 4 月帳單', BILL_AMT5:'前 5 月帳單', BILL_AMT6:'前 6 月帳單',
  PAY_AMT1:'最近月付款', PAY_AMT2:'前 2 月付款', PAY_AMT3:'前 3 月付款', PAY_AMT4:'前 4 月付款', PAY_AMT5:'前 5 月付款', PAY_AMT6:'前 6 月付款'
};

function valuesFromForm() {
  return [Number($('#limit').value), Number($('#age').value),
    ...[...$$('.delay-input')].map((el) => Number(el.value)),
    ...[...$$('.bill-input')].map((el) => Number(el.value)),
    ...[...$$('.paid-input')].map((el) => Number(el.value))];
}

$('#riskForm').addEventListener('submit', (event) => {
  event.preventDefault();
  const error = $('#formError');
  error.textContent = '';
  if (!model) { error.textContent = '模型仍在載入，請稍後再試。'; return; }
  if (!event.currentTarget.reportValidity()) return;
  const values = valuesFromForm();
  if (values.some((v) => !Number.isFinite(v))) { error.textContent = '請確認所有欄位均為有效數字。'; return; }
  const contributions = values.map((value, i) => model.coef[i] * ((value - model.mean[i]) / model.scale[i]));
  const logit = model.intercept + contributions.reduce((a, b) => a + b, 0);
  renderResult(sigmoid(logit), contributions);
});

function renderResult(probability, contributions) {
  const pct = Math.round(probability * 100);
  const high = probability >= model.threshold;
  $('#riskScore').textContent = `${pct}%`;
  $('#gauge').style.setProperty('--score', pct);
  const badge = $('#resultBadge');
  badge.className = `result-badge ${high ? 'high' : 'low'}`;
  badge.textContent = high ? '預測：較高違約風險' : '預測：較低違約風險';
  $('#riskText').textContent = high ? '建議進一步人工檢視' : '目前低於模型判定門檻';
  $('#resultExplain').textContent = `模型門檻為 ${Math.round(model.threshold * 100)}%。這是統計預測，不代表個人必然違約或不違約。`;
  const ranked = contributions.map((value, i) => ({ value, name: featureLabels[model.features[i]] }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 3);
  $('#drivers').innerHTML = ranked.map((item, i) => `<div><span>0${i+1}</span><b>${item.name}</b><em class="${item.value >= 0 ? 'up' : 'down'}">${item.value >= 0 ? '提高' : '降低'}風險</em></div>`).join('');
  renderInputTrend();
  $('.live-result').scrollIntoView({ behavior:'smooth', block:'center' });
}

function renderInputTrend() {
  const bills = [...$$('.bill-input')].map((el) => Number(el.value)).reverse();
  const paid = [...$$('.paid-input')].map((el) => Number(el.value)).reverse();
  const all = [...bills, ...paid];
  const low = Math.min(...all, 0);
  const high = Math.max(...all, 1);
  const points = (values) => values.map((value, i) => {
    const x = 10 + i * 80;
    const y = 100 - ((value - low) / (high - low || 1)) * 88;
    return `${x},${y.toFixed(1)}`;
  }).join(' ');
  $('#billTrend').setAttribute('points', points(bills));
  $('#paidTrend').setAttribute('points', points(paid));
}

