import { useState, useRef, useEffect, useCallback } from "react";

/* ─── responsive hook ──────────────────────────────────────── */
function useBreakpoint() {
  const [w, setW] = useState(window.innerWidth);
  useEffect(() => {
    const fn = () => setW(window.innerWidth);
    window.addEventListener("resize", fn);
    return () => window.removeEventListener("resize", fn);
  }, []);
  return { w, isMobile: w < 640, isTablet: w >= 640 && w < 1024, isDesktop: w >= 1024 };
}

/* ─── image processing ─────────────────────────────────────── */
function applyWarmth(ctx, w, h, amt) {
  if (!amt) return;
  const d = ctx.getImageData(0, 0, w, h), px = d.data;
  for (let i = 0; i < px.length; i += 4) {
    px[i]     = Math.min(255, px[i]     + amt * 1.2);
    px[i + 2] = Math.min(255, Math.max(0, px[i + 2] - amt * 0.8));
  }
  ctx.putImageData(d, 0, 0);
}
function applyPortraitTone(ctx, w, h) {
  const d = ctx.getImageData(0, 0, w, h), px = d.data;
  for (let i = 0; i < px.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      let v = px[i + c] / 255;
      if (v < 0.20) v = v + (0.20 - v) * 0.15;
      if (v > 0.88) v = 0.88 + (v - 0.88) * 0.70;
      v = Math.pow(v, 0.96);
      px[i + c] = Math.min(255, Math.max(0, Math.round(v * 255)));
    }
  }
  ctx.putImageData(d, 0, 0);
}
// ── Blur detection using Laplacian variance ──────────────────
// Low variance = blurry image, high variance = sharp image
function detectBlur(ctx, w, h) {
  const sample = ctx.getImageData(Math.floor(w*0.1), Math.floor(h*0.1), Math.floor(w*0.8), Math.floor(h*0.8));
  const px = sample.data, sw = Math.floor(w*0.8), sh = Math.floor(h*0.8);
  let sum = 0, count = 0;
  // Laplacian kernel: measures edge sharpness
  for (let y = 1; y < sh - 1; y += 2) {
    for (let x = 1; x < sw - 1; x += 2) {
      const i = (y * sw + x) * 4;
      const lum = (r,g,b) => r*0.299 + g*0.587 + b*0.114;
      const c  = lum(px[i],   px[i+1],   px[i+2]);
      const t  = lum(px[i-sw*4], px[i-sw*4+1], px[i-sw*4+2]);
      const b2 = lum(px[i+sw*4], px[i+sw*4+1], px[i+sw*4+2]);
      const l  = lum(px[i-4], px[i-3], px[i-2]);
      const r2 = lum(px[i+4], px[i+5], px[i+6]);
      const lap = Math.abs(4*c - t - b2 - l - r2);
      sum += lap * lap; count++;
    }
  }
  const variance = count > 0 ? sum / count : 9999;
  return { isBlurry: variance < 180, score: variance };
}

// ── Deblur pipeline — multi-pass sharpening + clarity + edge boost ──
function applyDeblur(ctx, w, h) {
  // Pass 1: strong unsharp mask
  applySharpen(ctx, w, h, 0.55);
  // Pass 2: edge enhancement using high-boost filter
  const src = ctx.getImageData(0, 0, w, h);
  const dst = ctx.createImageData(w, h);
  const s = src.data, d = dst.data;
  // High-boost kernel — amplifies edges more aggressively
  const K = [-1,-2,-1,-2,13,-2,-1,-2,-1];
  const blend = 0.35;
  for (let y = 1; y < h-1; y++) {
    for (let x = 1; x < w-1; x++) {
      const idx = (y*w+x)*4;
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let ky=-1; ky<=1; ky++)
          for (let kx=-1; kx<=1; kx++)
            v += s[((y+ky)*w+(x+kx))*4+c] * K[(ky+1)*3+(kx+1)];
        v /= 1;
        d[idx+c] = Math.min(255,Math.max(0,Math.round(s[idx+c]*(1-blend)+v*blend)));
      }
      d[idx+3] = s[idx+3];
    }
  }
  ctx.putImageData(dst, 0, 0);
  // Pass 3: clarity boost for micro-detail
  applyClarity(ctx, w, h);
  // Pass 4: final light sharpen to crisp up
  applySharpen(ctx, w, h, 0.22);
}

function isLowQuality(img) {
  return (img.width * img.height) / 1_000_000 < 4 || img.width < 2560 || img.height < 1440;
}
function upscaleCanvas(src, targetW, targetH) {
  let cur = document.createElement("canvas");
  cur.width = src.width; cur.height = src.height;
  cur.getContext("2d").drawImage(src, 0, 0);
  while (cur.width < targetW * 0.75 || cur.height < targetH * 0.75) {
    const next = document.createElement("canvas");
    next.width  = Math.min(Math.round(cur.width  * 1.5), targetW);
    next.height = Math.min(Math.round(cur.height * 1.5), targetH);
    const nc = next.getContext("2d");
    nc.imageSmoothingEnabled = true; nc.imageSmoothingQuality = "high";
    nc.drawImage(cur, 0, 0, next.width, next.height);
    cur = next;
  }
  const out = document.createElement("canvas");
  out.width = targetW; out.height = targetH;
  const oc = out.getContext("2d");
  oc.imageSmoothingEnabled = true; oc.imageSmoothingQuality = "high";
  oc.drawImage(cur, 0, 0, targetW, targetH);
  return out;
}
function needsSkinSmoothing(ctx, w, h) {
  const cx=Math.floor(w*0.25), cy=Math.floor(h*0.15);
  const sw=Math.floor(w*0.50), sh=Math.floor(h*0.55);
  const d=ctx.getImageData(cx,cy,sw,sh).data;
  let sum=0,sum2=0,cnt=0;
  for (let i=0;i<d.length;i+=4){
    const r=d[i],g=d[i+1],b=d[i+2];
    if(r>100&&r>g&&g>50&&b<r*0.85&&Math.abs(r-g)<80){
      const lum=r*0.299+g*0.587+b*0.114; sum+=lum; sum2+=lum*lum; cnt++;
    }
  }
  if(cnt<200) return true;
  return (sum2/cnt - (sum/cnt)**2) > 120;
}
function applySkinSmooth(ctx,w,h){
  const orig=ctx.getImageData(0,0,w,h),o=orig.data;
  const tmp=new Uint8ClampedArray(o.length),blr=new Uint8ClampedArray(o.length),R=3;
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    let sr=0,sg=0,sb=0,cnt=0;
    for(let kx=-R;kx<=R;kx++){const nx=Math.min(w-1,Math.max(0,x+kx)),ni=(y*w+nx)*4;sr+=o[ni];sg+=o[ni+1];sb+=o[ni+2];cnt++;}
    const ti=(y*w+x)*4; tmp[ti]=sr/cnt;tmp[ti+1]=sg/cnt;tmp[ti+2]=sb/cnt;tmp[ti+3]=o[ti+3];
  }
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    let sr=0,sg=0,sb=0,cnt=0;
    for(let ky=-R;ky<=R;ky++){const ny=Math.min(h-1,Math.max(0,y+ky)),ni=(ny*w+x)*4;sr+=tmp[ni];sg+=tmp[ni+1];sb+=tmp[ni+2];cnt++;}
    const bi=(y*w+x)*4; blr[bi]=sr/cnt;blr[bi+1]=sg/cnt;blr[bi+2]=sb/cnt;blr[bi+3]=o[bi+3];
  }
  const res=ctx.createImageData(w,h),rd=res.data;
  for(let i=0;i<o.length;i+=4){
    const lumO=o[i]*0.299+o[i+1]*0.587+o[i+2]*0.114;
    const lumB=blr[i]*0.299+blr[i+1]*0.587+blr[i+2]*0.114;
    const edge=Math.min(1,Math.abs(lumO-lumB)/18),keep=edge+(1-edge)*0.28;
    for(let c=0;c<3;c++) rd[i+c]=Math.min(255,Math.max(0,Math.round(o[i+c]*keep+blr[i+c]*(1-keep))));
    rd[i+3]=o[i+3];
  }
  ctx.putImageData(res,0,0);
}
function applySharpen(ctx,w,h,strength=0.28){
  const src=ctx.getImageData(0,0,w,h),dst=ctx.createImageData(w,h),s=src.data,d=dst.data;
  const K=[0,-1,0,-1,5,-1,0,-1,0];
  for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){
    const idx=(y*w+x)*4;
    for(let c=0;c<3;c++){
      let sh=0;
      for(let ky=-1;ky<=1;ky++) for(let kx=-1;kx<=1;kx++) sh+=s[((y+ky)*w+(x+kx))*4+c]*K[(ky+1)*3+(kx+1)];
      d[idx+c]=Math.min(255,Math.max(0,Math.round(s[idx+c]*(1-strength)+sh*strength)));
    }
    d[idx+3]=s[idx+3];
  }
  ctx.putImageData(dst,0,0);
}
function applyClarity(ctx,w,h){
  const src=ctx.getImageData(0,0,w,h),dst=ctx.createImageData(w,h),s=src.data,d=dst.data;
  const K=[-1,-1,-1,-1,-1,-1,2,2,2,-1,-1,2,8,2,-1,-1,2,2,2,-1,-1,-1,-1,-1,-1],blend=0.22;
  for(let y=2;y<h-2;y++) for(let x=2;x<w-2;x++){
    const idx=(y*w+x)*4;
    for(let c=0;c<3;c++){
      let v=0;
      for(let ky=-2;ky<=2;ky++) for(let kx=-2;kx<=2;kx++) v+=s[((y+ky)*w+(x+kx))*4+c]*K[(ky+2)*5+(kx+2)];
      v/=8; d[idx+c]=Math.min(255,Math.max(0,Math.round(s[idx+c]*(1-blend)+v*blend)));
    }
    d[idx+3]=s[idx+3];
  }
  ctx.putImageData(dst,0,0);
}
function applyUltraDetail(ctx,w,h){ applySharpen(ctx,w,h,0.45); applyClarity(ctx,w,h); applySharpen(ctx,w,h,0.30); applySharpen(ctx,w,h,0.18); }
function applyUltraFocus(ctx,w,h){
  applySharpen(ctx,w,h,0.55);
  const d=ctx.getImageData(0,0,w,h),px=d.data,cx=w/2,cy=h/2,maxD=Math.sqrt(cx*cx+cy*cy);
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const vig=1-Math.pow(Math.sqrt((x-cx)**2+(y-cy)**2)/maxD,2)*0.45,i=(y*w+x)*4;
    for(let c=0;c<3;c++) px[i+c]=Math.min(255,Math.max(0,Math.round(px[i+c]*vig)));
  }
  ctx.putImageData(d,0,0);
}

/* ─── splash screen ────────────────────────────────────────── */
function SplashScreen({ visible, fadingOut }) {
  return (
    <div style={{
      position:"fixed", inset:0, zIndex:9999,
      background:"#0d1b35",
      display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center",
      opacity: fadingOut ? 0 : 1,
      transition:"opacity 0.5s ease",
      pointerEvents: fadingOut ? "none" : "all",
    }}>
      <div style={{ marginBottom:32, position:"relative" }}>
        <div style={{
          width: "clamp(72px,15vw,96px)", height:"clamp(72px,15vw,96px)",
          borderRadius:"22%",
          background:"linear-gradient(135deg,#1a3560,#2a5298)",
          display:"flex", alignItems:"center", justifyContent:"center",
          boxShadow:"0 20px 60px rgba(26,53,96,0.6)",
          animation:"pulse 1.8s ease-in-out infinite",
        }}>
          <svg width="46%" height="46%" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
          </svg>
        </div>
        <div style={{
          position:"absolute", inset:"-14px",
          border:"2px solid rgba(255,255,255,0.12)",
          borderTopColor:"rgba(255,255,255,0.6)",
          borderRadius:"50%",
          animation:"spin 1.2s linear infinite",
        }}/>
      </div>
      <div style={{
        fontSize:"clamp(20px,5vw,30px)", fontWeight:800, color:"#fff",
        letterSpacing:"-0.03em", marginBottom:8, textAlign:"center", padding:"0 16px",
        animation:"fadeUp 0.6s ease 0.3s both",
      }}>Church Photo Studio</div>
      <div style={{
        fontSize:"clamp(10px,2.5vw,13px)", color:"rgba(255,255,255,0.45)",
        letterSpacing:"0.08em", textTransform:"uppercase",
        animation:"fadeUp 0.6s ease 0.5s both",
      }}>Media Enhancement Suite</div>
      <div style={{
        marginTop:44, width:"clamp(120px,40vw,180px)", height:3,
        background:"rgba(255,255,255,0.10)", borderRadius:4, overflow:"hidden",
        animation:"fadeUp 0.6s ease 0.7s both",
      }}>
        <div style={{
          height:"100%", borderRadius:4,
          background:"linear-gradient(90deg,#2a5298,#4a90d9)",
          animation:"progress 2.4s ease forwards",
        }}/>
      </div>
    </div>
  );
}

// Fast watermark composite — no heavy processing, just redraws wm on cached base
function compositeWatermark(baseDataUrl, watermarkImg, wmScale, wmOpacity, wmPos) {
  return new Promise(resolve => {
    if (!baseDataUrl) return resolve(null);
    const img = new Image();
    img.onload = () => {
      const W=img.width, H=img.height;
      const c=document.createElement("canvas"); c.width=W; c.height=H;
      const ctx=c.getContext("2d"); ctx.drawImage(img,0,0);
      if (watermarkImg) {
        const pad=W*0.025, wmW=W*(wmScale/100);
        const wmH=wmW/(watermarkImg.width/watermarkImg.height);
        let wx,wy;
        if(wmPos==="top-left"){wx=pad;wy=pad;}
        else if(wmPos==="top-center"){wx=(W-wmW)/2;wy=pad;}
        else if(wmPos==="top-right"){wx=W-wmW-pad;wy=pad;}
        else if(wmPos==="bottom-left"){wx=pad;wy=H-wmH-pad;}
        else if(wmPos==="bottom-right"){wx=W-wmW-pad;wy=H-wmH-pad;}
        else{wx=(W-wmW)/2;wy=H-wmH-pad;}
        ctx.globalAlpha=wmOpacity/100;
        ctx.drawImage(watermarkImg,wx,wy,wmW,wmH);
        ctx.globalAlpha=1;
      }
      resolve(c.toDataURL("image/jpeg",0.97));
    };
    img.src=baseDataUrl;
  });
}

/* ─── main app ─────────────────────────────────────────────── */
export default function App() {
  const { w, isMobile, isTablet } = useBreakpoint();
  const isSmall = isMobile || isTablet;

  const [loading,      setLoading]      = useState(true);
  const [profileOpen,  setProfileOpen]  = useState(false);
  const [churchName,   setChurchName]   = useState("Church Photo Studio");
  const [fadeOut,      setFadeOut]      = useState(false);
  const [watermark,    setWatermark]    = useState(null);
  const [watermarkImg, setWatermarkImg] = useState(null);
  const [profilePic,   setProfilePic]  = useState(null);
  const [queue,        setQueue]        = useState([]);
  const [dragging,     setDragging]     = useState(false);
  const [wmScale,      setWmScale]      = useState(35);
  const [wmOpacity,    setWmOpacity]    = useState(100);
  const [wmPos,        setWmPos]        = useState("bottom-center");

  const watermarkRef = useRef(null);
  const profileRef   = useRef(null);

  useEffect(() => {
    const t1 = setTimeout(() => setFadeOut(true), 2600);
    const t2 = setTimeout(() => setLoading(false), 3100);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  useEffect(() => {
    if (!watermark) { setWatermarkImg(null); return; }
    const img = new Image();
    img.onload = () => setWatermarkImg(img);
    img.src = watermark;
  }, [watermark]);

  const renderToCanvas = useCallback(async (src, applyFocus = false) => {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = async () => {
        const lowQ = isLowQuality(img);
        let oW=img.width, oH=img.height;
        const TARGET=4/5;
        let cW=oW, cH=oH, cX=0, cY=0;
        if(oW/oH>TARGET){cW=Math.round(oH*TARGET);cX=Math.round((oW-cW)/2);}
        else if(oW/oH<TARGET){cH=Math.round(oW/TARGET);}
        let W,H;
        if(lowQ){const scale=2048/cW;W=2048;H=Math.round(cH*scale);}
        else{W=cW;H=cH;}
        const off=document.createElement("canvas"); off.width=W; off.height=H;
        const oc=off.getContext("2d");
        if(lowQ){
          const cr=document.createElement("canvas"); cr.width=cW; cr.height=cH;
          cr.getContext("2d").drawImage(img,cX,cY,cW,cH,0,0,cW,cH);
          const up=upscaleCanvas(cr,W,H);
          oc.filter="brightness(1.06) contrast(1.06) saturate(1.14)";
          oc.drawImage(up,0,0,W,H);
        } else {
          oc.filter="brightness(1.05) contrast(1.04) saturate(1.10)";
          oc.drawImage(img,cX,cY,cW,cH,0,0,W,H);
        }
        oc.filter="none";
        applyPortraitTone(oc,W,H);
        if(needsSkinSmoothing(oc,W,H)) applySkinSmooth(oc,W,H);
        // Blur detection & deblur
        const blurResult = detectBlur(oc, W, H);
        const wasDeblurred = blurResult.isBlurry;
        if (wasDeblurred) {
          applyDeblur(oc, W, H);
        }

        applyWarmth(oc,W,H,3);
        if(lowQ && !wasDeblurred) applyUltraDetail(oc,W,H); else if(!wasDeblurred) applySharpen(oc,W,H,0.28);
        if(applyFocus) applyUltraFocus(oc,W,H);
        const canvas=document.createElement("canvas"); canvas.width=W; canvas.height=H;
        const ctx=canvas.getContext("2d"); ctx.drawImage(off,0,0);
        if(watermarkImg){
          const pad=W*0.025,wmW=W*(wmScale/100),wmH=wmW/(watermarkImg.width/watermarkImg.height);
          let wx,wy;
          if(wmPos==="top-left"){wx=pad;wy=pad;} else if(wmPos==="top-center"){wx=(W-wmW)/2;wy=pad;}
          else if(wmPos==="top-right"){wx=W-wmW-pad;wy=pad;} else if(wmPos==="bottom-left"){wx=pad;wy=H-wmH-pad;}
          else if(wmPos==="bottom-right"){wx=W-wmW-pad;wy=H-wmH-pad;} else{wx=(W-wmW)/2;wy=H-wmH-pad;}
          ctx.globalAlpha=wmOpacity/100; ctx.drawImage(watermarkImg,wx,wy,wmW,wmH); ctx.globalAlpha=1;
        }
        // Save base (no watermark) from off canvas
        const baseDataUrl=off.toDataURL("image/jpeg",0.97);
        // Composite watermark on top for final output
        if(watermarkImg){
          const pad=W*0.025,wmW=W*(wmScale/100),wmH=wmW/(watermarkImg.width/watermarkImg.height);
          let wx,wy;
          if(wmPos==="top-left"){wx=pad;wy=pad;} else if(wmPos==="top-center"){wx=(W-wmW)/2;wy=pad;}
          else if(wmPos==="top-right"){wx=W-wmW-pad;wy=pad;} else if(wmPos==="bottom-left"){wx=pad;wy=H-wmH-pad;}
          else if(wmPos==="bottom-right"){wx=W-wmW-pad;wy=H-wmH-pad;} else{wx=(W-wmW)/2;wy=H-wmH-pad;}
          ctx.globalAlpha=wmOpacity/100; ctx.drawImage(watermarkImg,wx,wy,wmW,wmH); ctx.globalAlpha=1;
        }
        resolve({dataUrl:canvas.toDataURL("image/jpeg",0.97), baseDataUrl, enhanced:lowQ, deblurred:wasDeblurred});
      };
      img.src=src;
    });
  }, [watermarkImg, wmScale, wmOpacity, wmPos]);

  const addPhotos = useCallback(async (files) => {
    const valid=Array.from(files).filter(f=>f.type.startsWith("image/"));
    if(!valid.length) return;
    const items=await Promise.all(valid.map(async file=>{
      const src=await new Promise(r=>{const rd=new FileReader();rd.onload=e=>r(e.target.result);rd.readAsDataURL(file);});
      return{id:`${Date.now()}-${Math.random()}`,name:file.name,original:src,status:"processing",output:null,focusOutput:null,focusActive:false,enhanced:false,deblurred:false};
    }));
    setQueue(prev=>[...prev,...items]);
    for(const item of items){
      const res=await renderToCanvas(item.original,false);
      setQueue(prev=>prev.map(q=>q.id===item.id?{...q,status:"done",output:res.dataUrl,baseOutput:res.baseDataUrl,enhanced:res.enhanced,deblurred:res.deblurred}:q));
    }
  }, [renderToCanvas]);

  // Full reprocess only when watermark image changes
  useEffect(()=>{
    if(!queue.length) return;
    const run=async()=>{
      for(const item of queue){
        setQueue(prev=>prev.map(q=>q.id===item.id?{...q,status:"processing"}:q));
        const res=await renderToCanvas(item.original,false);
        setQueue(prev=>prev.map(q=>q.id===item.id?{...q,status:"done",output:res.dataUrl,baseOutput:res.baseDataUrl,enhanced:res.enhanced,deblurred:res.deblurred}:q));
      }
    };
    run();
  },[watermarkImg]);

  // Fast path — only recomposites watermark on cached base (instant slider response)
  useEffect(()=>{
    if(!queue.length) return;
    let cancelled=false;
    const run=async()=>{
      for(const item of queue){
        if(cancelled||!item.baseOutput) continue;
        const output=await compositeWatermark(item.baseOutput,watermarkImg,wmScale,wmOpacity,wmPos);
        if(!cancelled) setQueue(prev=>prev.map(q=>q.id===item.id?{...q,output}:q));
      }
    };
    run();
    return()=>{cancelled=true;};
  },[wmScale,wmOpacity,wmPos]);

  const toggleFocus=useCallback(async(id)=>{
    const item=queue.find(q=>q.id===id);
    if(!item||item.status!=="done") return;
    const nowActive=!item.focusActive;
    setQueue(prev=>prev.map(q=>q.id===id?{...q,focusActive:nowActive,status:nowActive&&!item.focusOutput?"processing":q.status}:q));
    if(nowActive&&!item.focusOutput){
      const res=await renderToCanvas(item.original,true);
      setQueue(prev=>prev.map(q=>q.id===id?{...q,focusOutput:res.dataUrl,status:"done"}:q));
    } else {
      setQueue(prev=>prev.map(q=>q.id===id?{...q,status:"done"}:q));
    }
  },[queue,renderToCanvas]);

  const dl=(item)=>{
    const src=item.focusActive&&item.focusOutput?item.focusOutput:item.output;
    if(!src) return;
    try {
      const byteStr=atob(src.split(",")[1]);
      const mime=src.split(",")[0].split(":")[1].split(";")[0];
      const ab=new ArrayBuffer(byteStr.length);
      const ia=new Uint8Array(ab);
      for(let i=0;i<byteStr.length;i++) ia[i]=byteStr.charCodeAt(i);
      const blob=new Blob([ab],{type:mime});
      const url=URL.createObjectURL(blob);
      const a=document.createElement("a");
      a.href=url;
      a.download=`portrait-${item.name.replace(/\.[^.]+$/,"")}.jpg`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    } catch(e) {
      const w=window.open("","_blank");
      if(w) { w.document.write(`<img src="${src}" style="max-width:100%;display:block" />`); w.document.title="Save this image"; }
    }
  };
  const dlAll=()=>queue.filter(q=>q.status==="done").forEach((item,i)=>setTimeout(()=>dl(item),i*250));
  const doneCount=queue.filter(q=>q.status==="done").length;

  const POSITIONS=[
    {v:"top-left",l:"↖ Top left"},{v:"top-center",l:"↑ Top center"},{v:"top-right",l:"↗ Top right"},
    {v:"bottom-left",l:"↙ Bottom left"},{v:"bottom-center",l:"↓ Bottom center"},{v:"bottom-right",l:"↘ Bottom right"},
  ];

  const px = (d, m, s) => isMobile ? m : isTablet ? (s||m) : d;

  return (
    <>
      <style>{`
        *{box-sizing:border-box;margin:0;padding:0;}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
        @keyframes progress{from{width:0%}to{width:100%}}
        @keyframes pulse{0%,100%{transform:scale(1);box-shadow:0 20px 60px rgba(26,53,96,0.6)}50%{transform:scale(1.05);box-shadow:0 24px 70px rgba(26,53,96,0.8)}}
        @keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
        input[type=range]{width:100%;accent-color:#1a3560;}
        ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:#c0cfe0;border-radius:4px}
      `}</style>

      {/* ── Full Profile Page ── */}
      {profileOpen && (
        <div style={{
          position:"fixed",inset:0,zIndex:9000,background:"#f0f2f5",
          fontFamily:"'Segoe UI',system-ui,sans-serif",
          animation:"fadeIn 0.25s ease",overflowY:"auto",
        }}>
          {/* Topbar */}
          <div style={{
            background:"#fff",borderBottom:"1px solid #dde4ef",height:56,
            display:"flex",alignItems:"center",padding:"0 20px",gap:12,
            position:"sticky",top:0,zIndex:10,
          }}>
            <button onClick={()=>setProfileOpen(false)} style={{
              width:36,height:36,borderRadius:"50%",border:"1.5px solid #d0daea",
              background:"#f8fafd",cursor:"pointer",display:"flex",
              alignItems:"center",justifyContent:"center",flexShrink:0,
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1a3560" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 18 9 12 15 6"/>
              </svg>
            </button>
            <div style={{fontWeight:700,fontSize:15,color:"#111"}}>Edit Church Profile</div>
          </div>

          {/* Hero banner */}
          <div style={{background:"linear-gradient(135deg,#0d1b35 0%,#1a3560 100%)",height:160,position:"relative",overflow:"hidden"}}>
            <div style={{position:"absolute",top:-40,right:-40,width:180,height:180,borderRadius:"50%",background:"rgba(255,255,255,0.04)"}}/>
            <div style={{position:"absolute",bottom:-60,left:-20,width:200,height:200,borderRadius:"50%",background:"rgba(255,255,255,0.03)"}}/>
          </div>

          {/* Profile section */}
          <div style={{display:"flex",flexDirection:"column",alignItems:"center",marginTop:-70,padding:"0 20px"}}>
            {/* Profile picture */}
            <div style={{position:"relative",marginBottom:12}}>
              <div style={{
                width:130,height:130,borderRadius:"50%",
                background:profilePic?"transparent":"#1a3560",
                border:"5px solid #fff",overflow:"hidden",
                boxShadow:"0 8px 30px rgba(0,0,0,0.18)",
              }}>
                {profilePic
                  ? <img src={profilePic} alt="" style={{width:"100%",height:"100%",objectFit:"cover"}}/>
                  : <div style={{width:"100%",height:"100%",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:6}}>
                      <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                      </svg>
                      <span style={{fontSize:10,color:"rgba(255,255,255,0.5)"}}>No photo</span>
                    </div>
                }
              </div>
              {/* Camera button */}
              <div style={{position:"relative"}}>
                <div style={{
                  position:"absolute",bottom:4,right:4,width:34,height:34,
                  borderRadius:"50%",background:"#1a3560",border:"3px solid #fff",
                  display:"flex",alignItems:"center",justifyContent:"center",
                  cursor:"pointer",boxShadow:"0 2px 10px rgba(0,0,0,0.25)",
                }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                    <circle cx="12" cy="13" r="4"/>
                  </svg>
                </div>
                <input type="file" accept="image/*"
                  onChange={e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=ev=>setProfilePic(ev.target.result);r.readAsDataURL(f);}}
                  style={{position:"absolute",bottom:4,right:4,width:34,height:34,opacity:0,cursor:"pointer",zIndex:5}}
                />
              </div>
            </div>

            <div style={{fontSize:12,color:"#aaa",marginBottom:24}}>Tap the camera icon to change photo</div>

            {/* Form card */}
            <div style={{
              width:"100%",maxWidth:480,background:"#fff",borderRadius:16,
              border:"1px solid #dde4ef",padding:24,boxSizing:"border-box",marginBottom:32,
            }}>
              <div style={{fontSize:13,fontWeight:700,color:"#1a3560",marginBottom:18}}>Church Details</div>
              <div style={{marginBottom:16}}>
                <div style={{fontSize:11,color:"#888",fontWeight:600,marginBottom:6,textTransform:"uppercase",letterSpacing:"0.05em"}}>Church Name</div>
                <input value={churchName} onChange={e=>setChurchName(e.target.value)}
                  placeholder="e.g. Living Faith Church Woji"
                  style={{
                    width:"100%",padding:"12px 14px",border:"1.5px solid #d0daea",
                    borderRadius:9,fontSize:14,color:"#111",outline:"none",
                    background:"#f8fafd",boxSizing:"border-box",fontFamily:"inherit",
                  }}
                />
              </div>
              <button onClick={()=>setProfileOpen(false)} style={{
                width:"100%",background:"#1a3560",color:"#fff",border:"none",
                borderRadius:10,padding:13,fontSize:14,fontWeight:700,cursor:"pointer",
              }}>Save & Continue</button>
            </div>
          </div>
        </div>
      )}

      {loading && <SplashScreen fadingOut={fadeOut} />}

      <div style={{
        minHeight:"100vh", background:"#f0f2f5",
        fontFamily:"'Segoe UI',system-ui,sans-serif",
        opacity:loading?0:1, transition:"opacity 0.4s ease 0.1s",
      }}>

        {/* ── Topbar ── */}
        <div style={{
          background:"#fff", borderBottom:"1px solid #dde4ef",
          padding:`0 ${px(28,14,20)}px`, height:px(58,52,56),
          display:"flex", alignItems:"center", gap:px(12,8,10),
          position:"sticky", top:0, zIndex:100,
        }}>
          <div style={{position:"relative",flexShrink:0,marginRight:4}}>
            {/* Circular profile picture */}
            <div style={{
              width:52, height:52, borderRadius:"50%",
              background:profilePic?"transparent":"#1a3560",
              border:"3px solid #1a3560",
              display:"flex",alignItems:"center",justifyContent:"center",
              overflow:"hidden",
              boxShadow:"0 2px 12px rgba(26,53,96,0.25)",
            }}>
              {profilePic
                ?<img src={profilePic} alt="" style={{width:"100%",height:"100%",objectFit:"cover"}}/>
                :<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                  <circle cx="12" cy="13" r="4"/>
                </svg>
              }
            </div>
            {/* Edit badge — opens profile modal */}
            <div onClick={()=>setProfileOpen(true)} style={{
              position:"absolute",bottom:-4,left:"50%",transform:"translateX(-50%)",
              background:"#1a3560",borderRadius:20,padding:"2px 8px",
              display:"flex",alignItems:"center",gap:3,
              border:"2px solid #fff",zIndex:4,whiteSpace:"nowrap",
              boxShadow:"0 1px 4px rgba(0,0,0,0.2)",cursor:"pointer",
            }}>
              <span style={{fontSize:9,color:"#fff",fontWeight:700,letterSpacing:"0.03em"}}>Edit</span>
            </div>
          </div>
          <div style={{flex:1,minWidth:0}}>
            <div style={{fontWeight:700,fontSize:px(14,12,13),color:"#111",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{churchName}</div>
            <div style={{fontSize:px(11,10,10),color:"#999",display:isMobile?"none":"block"}}>
              {profilePic?"Click photo to update church profile":"Click to upload church profile picture"}
            </div>
          </div>
          {doneCount>0&&(
            <button onClick={dlAll} style={{
              flexShrink:0, background:"#1a3560", color:"#fff",
              border:"none", borderRadius:8, padding:px("8px 18px","6px 10px","7px 14px"),
              fontSize:px(13,11,12), fontWeight:600, cursor:"pointer", whiteSpace:"nowrap",
            }}>↓ {isMobile?`(${doneCount})`:`Download all (${doneCount})`}</button>
          )}
        </div>

        <div style={{maxWidth:980,margin:"0 auto",padding:`${px(24,14,18)}px ${px(20,12,16)}px`}}>

          {/* ── Watermark card ── */}
          <div style={{background:"#fff",borderRadius:14,border:"1px solid #dde4ef",padding:px("22px 24px","14px 14px","18px 18px"),marginBottom:16}}>
            <div style={{fontWeight:700,fontSize:px(13,11,12),color:"#1a3560",marginBottom:px(18,12,14),display:"flex",alignItems:"center",gap:8}}>
              <span style={{
                width:20,height:20,borderRadius:"50%",background:watermarkImg?"#22a06b":"#1a3560",
                color:"#fff",fontSize:11,fontWeight:700,display:"inline-flex",alignItems:"center",justifyContent:"center",flexShrink:0,
              }}>{watermarkImg?"✓":"1"}</span>
              WATERMARK — upload once, applied to every photo
            </div>

            {/* Upload + sliders — stacks on mobile */}
            <div style={{
              display:"grid",
              gridTemplateColumns:isMobile?"1fr":isTablet?"1fr 1fr":"auto 1fr",
              gap:px(24,14,18), alignItems:"start", marginBottom:16,
            }}>
              <div>
                <div style={{fontSize:11,color:"#888",fontWeight:600,marginBottom:8}}>YOUR WATERMARK</div>
                <div style={{position:"relative",width:"100%",maxWidth:px(200,220,220),height:px(80,70,75)}}>
                  <div style={{
                    width:"100%",height:"100%",
                    border:watermark?"2px solid #1a3560":"2px dashed #b8c8dc",
                    borderRadius:10,background:watermark?"#f8fafd":"#fafafa",
                    display:"flex",alignItems:"center",justifyContent:"center",overflow:"hidden",
                  }}>
                    {watermark
                      ?<img src={watermark} alt="" style={{maxWidth:"100%",maxHeight:"100%",objectFit:"contain",padding:6}}/>
                      :<div style={{textAlign:"center"}}>
                        <div style={{fontSize:22,color:"#ccc",lineHeight:1}}>↑</div>
                        <div style={{fontSize:px(12,11,11),color:"#aaa",marginTop:4}}>Upload watermark</div>
                        <div style={{fontSize:10,color:"#bbb",marginTop:2,display:isMobile?"none":"block"}}>PNG with transparency works best</div>
                      </div>
                    }
                  </div>
                  <input type="file" accept="image/*"
                    onChange={e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=ev=>setWatermark(ev.target.result);r.readAsDataURL(f);}}
                    style={{position:"absolute",inset:0,opacity:0,cursor:"pointer",width:"100%",height:"100%"}}/>
                </div>
              </div>
              <div>
                <div style={{fontSize:11,color:"#888",fontWeight:600,marginBottom:12}}>WATERMARK SETTINGS</div>
                {[{label:"Size",val:wmScale,set:setWmScale,min:10,max:70,unit:"% width"},{label:"Opacity",val:wmOpacity,set:setWmOpacity,min:20,max:100,unit:"%"}].map(s=>(
                  <div key={s.label} style={{marginBottom:14}}>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:12,color:"#444",marginBottom:6}}>
                      <span>{s.label}</span><span style={{color:"#1a3560",fontWeight:600}}>{s.val}{s.unit}</span>
                    </div>
                    <input type="range" min={s.min} max={s.max} step={1} value={s.val} onChange={e=>s.set(Number(e.target.value))}/>
                  </div>
                ))}
              </div>
            </div>

            {/* Placement grid */}
            <div>
              <div style={{fontSize:11,color:"#888",fontWeight:600,marginBottom:10}}>PLACEMENT</div>
              <div style={{display:"grid",gridTemplateColumns:isMobile?"1fr 1fr":"1fr 1fr 1fr",gap:8}}>
                {POSITIONS.map(p=>(
                  <button key={p.v} onClick={()=>setWmPos(p.v)}
                    onMouseEnter={e=>{if(wmPos!==p.v){e.currentTarget.style.background="#eef2f9";e.currentTarget.style.borderColor="#1a3560";e.currentTarget.style.color="#1a3560";}}}
                    onMouseLeave={e=>{if(wmPos!==p.v){e.currentTarget.style.background="#f8fafd";e.currentTarget.style.borderColor="#d0daea";e.currentTarget.style.color="#555";}}}
                    style={{
                      padding:px("9px 8px","7px 4px","8px 6px"),fontSize:px(12,11,11),
                      borderRadius:8,cursor:"pointer",textAlign:"center",transition:"all 0.15s",
                      border:wmPos===p.v?"2px solid #1a3560":"1.5px solid #d0daea",
                      background:wmPos===p.v?"#eef2f9":"#f8fafd",
                      color:wmPos===p.v?"#1a3560":"#555",
                      fontWeight:wmPos===p.v?600:400,
                    }}>{p.l}</button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Upload zone ── */}
          <div style={{position:"relative",marginBottom:20}}>
            <div
              onDragOver={e=>{e.preventDefault();setDragging(true);}}
              onDragLeave={()=>setDragging(false)}
              onDrop={e=>{e.preventDefault();setDragging(false);addPhotos(e.dataTransfer.files);}}
              style={{
                border:`2.5px dashed ${dragging?"#1a3560":"#a0b8d4"}`,
                borderRadius:14, padding:px("36px 24px","24px 16px","30px 20px"), textAlign:"center",
                background:dragging?"#eef2f9":"#f8fafd", transition:"all 0.18s", cursor:"pointer",
              }}
            >
              <div style={{
                width:px(56,44,50),height:px(56,44,50),borderRadius:14,background:"#eef2f9",
                display:"flex",alignItems:"center",justifyContent:"center",margin:"0 auto 12px",pointerEvents:"none",
              }}>
                <svg width="52%" height="52%" viewBox="0 0 24 24" fill="none" stroke="#1a3560" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
                </svg>
              </div>
              <div style={{fontWeight:700,fontSize:px(16,14,15),color:"#1a1a2e",marginBottom:6,pointerEvents:"none"}}>Upload Church Photos</div>
              <div style={{fontSize:px(13,11,12),color:"#aaa",pointerEvents:"none"}}>
                {isMobile?"Tap to select photos":"Drag & drop or click · Multiple photos · Enhancements applied automatically"}
              </div>
            </div>
            <input type="file" accept="image/*" multiple onChange={e=>addPhotos(e.target.files)}
              style={{position:"absolute",inset:0,width:"100%",height:"100%",opacity:0,cursor:"pointer",zIndex:5}}/>
          </div>

          {/* ── Photo grid ── */}
          {queue.length>0&&(
            <>
              <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:14,flexWrap:"wrap",gap:8}}>
                <div style={{fontWeight:700,fontSize:px(14,12,13),color:"#1a1a2e"}}>
                  {doneCount} of {queue.length} photo{queue.length>1?"s":""} ready
                </div>
                <div style={{display:"flex",gap:8}}>
                  <div style={{position:"relative",display:"inline-block"}}>
                    <button style={{
                      background:"transparent",border:"1.5px solid #d0daea",borderRadius:7,
                      padding:px("6px 14px","5px 10px","6px 12px"),fontSize:px(12,11,12),
                      color:"#1a3560",fontWeight:600,cursor:"pointer",pointerEvents:"none",
                    }}>+ Add more</button>
                    <input type="file" accept="image/*" multiple onChange={e=>addPhotos(e.target.files)}
                      style={{position:"absolute",inset:0,opacity:0,cursor:"pointer",width:"100%",height:"100%"}}/>
                  </div>
                  <button onClick={()=>setQueue([])} style={{
                    background:"transparent",border:"1.5px solid #fde8e8",borderRadius:7,
                    padding:px("6px 14px","5px 10px","6px 12px"),fontSize:px(12,11,12),
                    color:"#c0392b",cursor:"pointer",fontWeight:600,
                  }}>Clear all</button>
                </div>
              </div>

              <div style={{
                display:"grid",
                gridTemplateColumns:`repeat(auto-fill,minmax(${px(200,140,170)}px,1fr))`,
                gap:px(16,10,12),
              }}>
                {queue.map(item=>{
                  const displaySrc=item.focusActive&&item.focusOutput?item.focusOutput:item.output;
                  return(
                    <div key={item.id} style={{
                      background:"#fff",borderRadius:12,border:"1px solid #e4e8ef",
                      overflow:"hidden",animation:"fadeIn 0.3s ease",
                    }}>
                      <div style={{position:"relative",aspectRatio:"4/5",background:"#eef2f9",overflow:"hidden"}}>
                        {item.status==="processing"&&(
                          <div style={{position:"absolute",inset:0,zIndex:2,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",background:"rgba(248,250,253,0.88)"}}>
                            <div style={{width:34,height:34,borderRadius:"50%",border:"3px solid #d0daea",borderTopColor:"#1a3560",animation:"spin 0.7s linear infinite",marginBottom:8}}/>
                            <div style={{fontSize:11,color:"#1a3560",fontWeight:600}}>Enhancing…</div>
                          </div>
                        )}
                        <img src={displaySrc||item.original} alt={item.name}
                          style={{width:"100%",height:"100%",objectFit:"cover",display:"block",opacity:displaySrc?1:0.25,transition:"opacity 0.3s"}}/>
                        <button onClick={()=>setQueue(prev=>prev.filter(q=>q.id!==item.id))} style={{
                          position:"absolute",top:6,right:6,width:24,height:24,borderRadius:"50%",
                          background:"rgba(0,0,0,0.5)",border:"none",color:"#fff",fontSize:15,
                          cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",lineHeight:1,
                        }}>×</button>
                        {item.status==="done"&&(
                          <div style={{position:"absolute",bottom:6,left:6,display:"flex",gap:4,flexWrap:"wrap"}}>
                            <div style={{background:"rgba(0,0,0,0.45)",borderRadius:20,padding:"2px 7px",fontSize:px(9,8,8),color:"#fff",fontWeight:700}}>4:5</div>
                            {item.enhanced&&<div style={{background:"rgba(26,53,96,0.85)",borderRadius:20,padding:"2px 7px",fontSize:px(9,8,8),color:"#fff",fontWeight:700}}>⬆ 4K</div>}
                            {item.deblurred&&<div style={{background:"rgba(16,100,60,0.88)",borderRadius:20,padding:"2px 7px",fontSize:px(9,8,8),color:"#fff",fontWeight:700}}>✦ DEBLURRED</div>}
                          </div>
                        )}
                      </div>
                      <div style={{padding:px("10px 12px","8px 10px","9px 11px")}}>
                        <div style={{fontSize:px(11,10,11),fontWeight:600,color:"#111",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",marginBottom:8}}>{item.name}</div>
                        {item.status==="done"&&(
                          <button onClick={()=>toggleFocus(item.id)} style={{
                            width:"100%",marginBottom:8,padding:px("7px","5px","6px"),fontSize:px(11,10,11),
                            borderRadius:7,cursor:"pointer",fontWeight:600,transition:"all 0.15s",
                            border:item.focusActive?"2px solid #1a3560":"1.5px solid #d0daea",
                            background:item.focusActive?"#1a3560":"#f8fafd",
                            color:item.focusActive?"#fff":"#1a3560",
                          }}>
                            {item.focusActive?"✓ Ultra-focus ON":"+ Ultra-focus portrait"}
                          </button>
                        )}
                        <div style={{display:"flex",alignItems:"center",gap:8}}>
                          <div style={{flex:1,fontSize:10,color:item.status==="done"?"#22a06b":"#aaa",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                            {item.status==="done"?item.focusActive?"✓ Ultra-focus applied":"✓ Enhanced":"Processing…"}
                          </div>
                          {item.status==="done"&&(
                            <button onClick={()=>dl(item)} style={{
                              background:"#1a3560",color:"#fff",border:"none",borderRadius:7,
                              padding:px("6px 14px","5px 10px","6px 12px"),fontSize:px(12,11,12),
                              fontWeight:600,cursor:"pointer",flexShrink:0,
                            }}>↓ Save</button>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
