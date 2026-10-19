// Requires playwright-core and an isolated Chrome executable; localhost only.
const {chromium} = require('playwright-core');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const base = process.env.WINTAK_DEMO_URL || 'http://127.0.0.1:8766';
const output = process.env.WINTAK_QA_DIR || __dirname;
const result = {created_utc: new Date().toISOString(), checks: [], page_errors: [], local_http_errors: [], blocked_external: []};
(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.WINTAK_BROWSER_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 900}});
    page.setDefaultTimeout(20000);
    page.on('pageerror', e => result.page_errors.push(e.message));
    page.on('response', r => {if (r.url().startsWith(base) && r.status() >= 400) result.local_http_errors.push([r.url(), r.status()]);});
    await page.route('**/*', route => {
      const u = new URL(route.request().url());
      if (u.origin === base || ['data:', 'blob:'].includes(u.protocol)) return route.continue();
      result.blocked_external.push(u.origin); return route.abort();
    });
    const response = await page.goto(base + '/corridor_handoff_monitor.html', {waitUntil: 'domcontentloaded'});
    assert.match(response.headers()['content-type'], /charset=utf-8/);
    assert.equal(await page.evaluate(() => document.characterSet), 'UTF-8');
    result.checks.push({name: 'UTF-8 HTML delivery', passed: true});
    await page.waitForFunction(() => document.querySelectorAll('#dock video').length === 4 && [...document.querySelectorAll('#dock video')].every(v => v.readyState >= 2));
    const videos = await page.locator('#dock video').evaluateAll(vs => vs.map(v => ({src: v.getAttribute('src'), duration: v.duration, width: v.videoWidth, error: v.error?.message || null})));
    assert.ok(videos.every(v => v.duration === 600 && v.width === 640 && !v.error));
    result.checks.push({name: 'Four actual video feeds decode', passed: true, evidence: videos});
    await page.locator('#scrub').evaluate(e => {e.value = '300'; e.dispatchEvent(new Event('input', {bubbles: true}));});
    await page.waitForFunction(() => [...document.querySelectorAll('#dock video')].every(v => !v.seeking && Math.abs(v.currentTime - 300) < 0.15));
    result.checks.push({name: 'Four-camera seek to 300 seconds', passed: true});
    await page.locator('#play').click(); await page.waitForFunction(() => state.t > 301); await page.locator('#play').click();
    assert.equal(await page.locator('#play').getAttribute('aria-pressed'), 'false');
    result.checks.push({name: 'Play and pause advance replay', passed: true});
    await page.locator('#tabAnalysis').click(); assert.ok(await page.locator('#paneAnalysis').isVisible());
    assert.ok((await page.locator('#kpi').innerText()).length > 0);
    await page.locator('#tabSensors').click(); assert.ok(await page.locator('#paneSensors').isVisible());
    await page.locator('#tabAlerts').click(); await page.locator('#zin').click(); await page.locator('#zfit').click();
    result.checks.push({name: 'Analysis, sensor/alert tabs and map controls', passed: true});
    result.layout = await page.evaluate(() => ({width: innerWidth, height: innerHeight, scrollWidth: document.documentElement.scrollWidth, scrollHeight: document.documentElement.scrollHeight, bodyMargin: getComputedStyle(document.body).margin}));
    result.known_issues = result.layout.scrollHeight > result.layout.height ? ['Original page retains body margin and document extends below viewport; application source preserved.'] : [];
    await page.screenshot({path: path.join(output, 'corridor.png'), fullPage: true});
    await page.goto(base + '/perimeter_handoff_monitor.html', {waitUntil: 'domcontentloaded'});
    assert.equal(await page.locator('#map').count(), 1);
    result.checks.push({name: 'Synthetic tactical page loads', passed: true});
    assert.deepEqual(result.page_errors, []); assert.deepEqual(result.local_http_errors, []);
    result.checks.push({name: 'No application errors or failed local resources', passed: true});
    result.functional_passed = true;
  } catch (error) {
    result.functional_passed = false; result.error = error.stack; process.exitCode = 1;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(output, 'checkpoint.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
  }
})();
