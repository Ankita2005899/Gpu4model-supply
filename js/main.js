// Edit "link" to point to your real tool page / Colab / app URL.
const TOOLS=[
 {id:"text-to-video-1",icon:"🎬",name:"Text to Video 1",desc:"Turn a prompt into a short video clip.",link:"https://colab.research.google.com/github/Ankita2005899/Gpu4model-supply/blob/main/notebooks/ai-avatar-narrator.ipynb"},
 {id:"text-to-video-2",icon:"🎞️",name:"Text to Video 2",desc:"Second video model for a different style.",link:"#"},
 {id:"video-to-script",icon:"📝",name:"Video to Script",desc:"Get a clean script from any video.",link:"https://colab.research.google.com/github/Ankita2005899/Gpu4model-supply/blob/main/notebooks/scriptsync-colab.ipynb"},
 {id:"text-to-speech",icon:"🔊",name:"Text to Speech",desc:"Natural voice from your text.",link:"https://colab.research.google.com/github/Ankita2005899/Gpu4model-supply/blob/main/notebooks/bolchehra-talking-face.ipynb"},
 {id:"background-remover",icon:"✂️",name:"Background Remover",desc:"Remove backgrounds from images and video.",link:"#"},
 {id:"voice-ai-assistant",icon:"🎙️",name:"Voice AI Assistant",desc:"Talk to your AI and get spoken answers.",link:"#"}
];
const grid=document.getElementById("grid");
TOOLS.forEach(t=>{
 const c=document.createElement("article");c.className="card";
 c.innerHTML=`<h2><span class="ic">${t.icon}</span>${t.name}</h2><p>${t.desc}</p>
 <a class="btn" href="${t.link}" target="_blank" rel="noopener">Open ${t.name}</a>
 <div class="vid"><video controls preload="metadata" src="videos/${t.id}.mp4"></video><span class="ph">Demo video goes here<br>videos/${t.id}.mp4</span></div>
 <label class="up" tabindex="0">Preview a video from your device<input type="file" accept="video/*"></label>`;
 const v=c.querySelector("video"),box=c.querySelector(".vid");
 v.addEventListener("loadeddata",()=>box.classList.add("has"));
 c.querySelector("input").addEventListener("change",e=>{const f=e.target.files[0];if(f){v.src=URL.createObjectURL(f);box.classList.add("has")}});
 grid.appendChild(c);
});
const cv=document.getElementById("bg"),x=cv.getContext("2d");let W,H,P=[];
function rs(){W=cv.width=innerWidth;H=cv.height=innerHeight;P=Array.from({length:Math.min(70,W/18)},()=>({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4}))}
rs();addEventListener("resize",rs);
(function f(){x.clearRect(0,0,W,H);
 P.forEach((a,i)=>{a.x+=a.vx;a.y+=a.vy;if(a.x<0||a.x>W)a.vx*=-1;if(a.y<0||a.y>H)a.vy*=-1;
  x.fillStyle="#22d3ee";x.fillRect(a.x,a.y,2,2);
  for(let j=i+1;j<P.length;j++){const b=P[j],d=Math.hypot(a.x-b.x,a.y-b.y);
   if(d<130){x.strokeStyle=`rgba(168,85,247,${.25*(1-d/130)})`;x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}}});
 if(!matchMedia("(prefers-reduced-motion:reduce)").matches)requestAnimationFrame(f)})();
