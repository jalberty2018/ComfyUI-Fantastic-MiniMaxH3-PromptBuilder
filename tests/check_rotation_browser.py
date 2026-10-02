"""Exercise the real trim editor in headless Edge without a ComfyUI server.

Run: py tests/check_rotation_browser.py
Only ComfyUI's app/API boundary is stubbed; editor code and CSS are unchanged.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
SOURCE = (ROOT / "web/medialoader.js").read_text(encoding="utf-8")
SOURCE = re.sub(r'^import .*?;\s*', '', SOURCE, flags=re.M)
SOURCE = re.sub(r'\bexport (?=(?:const|function|async|class)\b)', '', SOURCE)
STUB = """
let mediaURL = '';
const app = {registerExtension() {}, canvas: {setDirty() {}}};
const api = {apiURL() { return mediaURL; }};
"""
TEST = r"""
const result = document.createElement('pre'); result.id = 'result'; document.body.append(result);
let checks = 0;
function check(ok, message) { if (!ok) throw Error(message); checks++; }
function near(a, b) { return Math.abs(a-b) < 1.1; }
const settle = () => new Promise(r => setTimeout(r, 60));
function geometry(m) {
  const s=m.stage.getBoundingClientRect(), p=m.media.getBoundingClientRect(), c=m.cropBox.getBoundingClientRect();
  check(p.width > 0 && p.height > 0, 'Preview has dimensions');
  check(p.left >= s.left-1 && p.right <= s.right+1 && p.top >= s.top-1 && p.bottom <= s.bottom+1, 'Preview stays inside stage '+JSON.stringify({s:s.toJSON(),p:p.toJSON(),rotate:m.rotate,style:m.media.style.cssText}));
  check(['left','top','width','height'].every(k=>near(p[k],c[k])), 'Crop overlay follows painted preview');
  const b=m.rotBtn.getBoundingClientRect();
  check(b.bottom <= s.top+1 || b.top >= s.bottom-1, 'Rotation toolbar does not overlap preview');
  check(m.modal.scrollWidth <= m.modal.clientWidth+1, 'No horizontal toolbar overflow');
}
(async () => {
  const panel = {items: [], commit() {}, say() {}, render() {}};
  for (const [w,h] of [[1600,900],[900,1600],[1000,1000]]) {
    const cv=document.createElement('canvas'); cv.width=w; cv.height=h;
    const g=cv.getContext('2d'); g.fillStyle='#278bb5'; g.fillRect(0,0,w,h);
    g.fillStyle='#ffcc44'; g.fillRect(0,0,w/2,h/2); mediaURL=cv.toDataURL();
    const item={kind:'picture', file:'test.png', name:`${w}x${h}`, width:w,height:h,crop:{x:.1,y:.2,w:.5,h:.4}};
    const m=new TrimModal(panel,item); await m.media.decode(); await settle();
    const original=JSON.stringify(m.crop);
    m.cropMode=true; m.syncCrop();
    const rect=m.cropRect.getBoundingClientRect();
    m.cropRect.dispatchEvent(new MouseEvent('mousedown',{bubbles:true,clientX:rect.left+10,clientY:rect.top+10}));
    window.dispatchEvent(new MouseEvent('mousemove',{clientX:rect.left+30,clientY:rect.top+20}));
    window.dispatchEvent(new MouseEvent('mouseup'));
    check(m.crop.x>.1 && m.crop.y>.2,'Crop can be dragged on the preview');
    m.crop=JSON.parse(original);m.syncCrop();
    for(const size of [.6,1,2]) {
      saveEditorSize({window:size,text:1}); applyEditorSize(m.modal);
      await settle();
      for(let i=0;i<4;i++) {
        geometry(m);
        check(item.width===w && item.height===h,'Source metadata stays immutable');
        const buttonWidth=m.rotBtn.getBoundingClientRect().width;
        m.rotBtn.click(); await settle();
        check(near(buttonWidth,m.rotBtn.getBoundingClientRect().width),'Rotation button has stable width');
      }
      check(Object.keys(m.crop).every(k=>Math.abs(m.crop[k]-JSON.parse(original)[k])<1e-9),'Four turns restore crop');
    }
    m.rotBtn.dispatchEvent(new MouseEvent('click',{shiftKey:true})); await settle();
    check(m.rotate===270,'Shift click rotates anticlockwise'); geometry(m);
    m.mirrorBtn.click(); await settle(); geometry(m);
    m.apply(); check(item.rotate===270 && item.mirror===true,'Apply persists rotation and mirror');
    const reopened=new TrimModal(panel,item); await reopened.media.decode(); await settle(); geometry(reopened); reopened.close();
    const locked=new TrimModal(panel,item,{refmod:true,aspect:1}); await locked.media.decode(); await settle();
    locked.rotBtn.click(); await settle(); geometry(locked);
    const [vw,vh]=locked.visualSize();
    check(near(locked.crop.w*vw,locked.crop.h*vh),'RefMod locked aspect survives rotation'); locked.close();
  }
  saveEditorSize({window:1,text:1});
  mediaURL='';
  const video=new TrimModal(panel,{kind:'video',file:'test.mp4',name:'Mask layout',width:1920,height:1080,duration:5,crop:{x:.1,y:.1,w:.8,h:.8}});
  await settle(); geometry(video);
  video.setMaskMode(true); await settle(); geometry(video);
  check(video.masker && !video.maskSide.hidden,'Mask editor opens beside preview');
  check(video.masker.ckpt.value===SAM_AUTO,'Missing SAM defaults to automatic download');
  let queued=null;
  api.fetchApi=async (path,opts)=>{queued=JSON.parse(opts.body);return {ok:false,json:async()=>({error:'Test stops before GPU execution'})};};
  await video.masker.run({id:'test-auto',text:'person',marks:[]});
  check(queued.prompt['1'].class_type==='MiniMaxH3SAMLoader','Auto Mask queues the download-capable loader');
  check(queued.prompt['1'].inputs.ckpt_name===SAM_AUTO,'Auto choice reaches loader');
  video.setMaskMode(false); await settle(); geometry(video); video.close();
  result.textContent=JSON.stringify({ok:true,checks});
})().catch(e=>{result.textContent=JSON.stringify({ok:false,checks,error:e.stack});});
"""

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="fh3-rotation-") as tmp:
        page = Path(tmp) / "test.html"
        page.write_text('<!doctype html><meta charset="utf-8"><body><script>' + (STUB + SOURCE + TEST).replace('</script', '<\\/script') + '</script>', encoding="utf-8")
        browser = subprocess.Popen([str(EDGE), '--headless=new', '--disable-gpu', '--no-first-run',
                                    '--no-default-browser-check', '--disable-extensions',
                                    '--allow-file-access-from-files', '--window-size=1280,1000',
                                    '--remote-debugging-port=0',
                                    '--user-data-dir=' + str(Path(tmp) / 'profile'), 'about:blank'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            port_file = Path(tmp) / 'profile/DevToolsActivePort'
            for _ in range(100):
                try:
                    port = port_file.read_text().splitlines()[0]
                    break
                except (OSError, IndexError):
                    time.sleep(.1)
            else:
                raise RuntimeError('Edge did not open its debugging port')
            runner = r"""
const [port,url] = process.argv.slice(1);
const tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const ws = new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0;const pending=new Map();
ws.addEventListener('message',e=>{const v=JSON.parse(e.data);if(pending.has(v.id)){pending.get(v.id)(v);pending.delete(v.id);}});
const call=(method,params={})=>new Promise(r=>{pending.set(++id,r);ws.send(JSON.stringify({id,method,params}));});
await call('Runtime.enable');
ws.addEventListener('message',e=>{const v=JSON.parse(e.data);if(v.method==='Runtime.exceptionThrown')console.error(JSON.stringify(v.params));});
await call('Page.navigate',{url});
let report;
for(let i=0;i<250;i++){
 await new Promise(r=>setTimeout(r,100));
 const v=await call('Runtime.evaluate',{expression:'document.getElementById("result")?.textContent',returnByValue:true});
 if(v.result?.result?.value){report=v.result.result.value;break;}
}
if(!report)console.error(JSON.stringify(await call('Runtime.evaluate',{expression:'JSON.stringify({url:location.href,state:document.readyState,result:document.getElementById("result")?.outerHTML,body:document.body.innerText.slice(-1000)})',returnByValue:true})));
console.log(report||'No browser test result');
ws.close();
process.exitCode=report && JSON.parse(report).ok ? 0 : 1;
"""
            node = shutil.which('node') or 'C:/Program Files/nodejs/node.exe'
            run = subprocess.run([node, '--input-type=module', '-e', runner, port, page.as_uri()],
                                 capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=40)
            print(run.stdout)
            if run.returncode: print(run.stderr)
            if run.returncode:
                raise SystemExit(run.returncode)
        finally:
            browser.terminate()
            browser.wait(timeout=10)
