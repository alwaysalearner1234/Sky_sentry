import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { Lensflare, LensflareElement } from 'three/addons/objects/Lensflare.js';

const CONTAINER_ID = 'earth-canvas-container';

const TEX = {
  diffuse: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/planets/earth_atmos_2048.jpg',
  normal: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/planets/earth_normal_2048.jpg',
  specular: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/planets/earth_specular_2048.jpg',
  clouds: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/planets/earth_clouds_1024.png',
  flare: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/lensflare/lensflare0.png',
  flare2: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/lensflare/lensflare2.png',
};

const TLES = [
  { name: "GOES-16", line1: "1 41866U 16071A   26289.50000000  .00000000  00000+0  00000+0 0  9999", line2: "2 41866   0.0400  85.0000 0001000  90.0000 270.0000  1.00270000 12345", period: 1436 }, // minutes
  { name: "NOAA-20", line1: "1 43013U 17073A   26289.50000000  .00000100  00000+0  10000-3 0  9999", line2: "2 43013  98.7000 120.0000 0001000  90.0000 270.0000 14.19500000450000", period: 101 } // minutes
];

const container = document.getElementById(CONTAINER_ID);
if (container) initEarth(container).catch(console.error);

function latLonToVec3(lat, lon, radius) {
  const phi = (90 - lat) * (Math.PI / 180); const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(-radius * Math.sin(phi) * Math.cos(theta), radius * Math.cos(phi), radius * Math.sin(phi) * Math.sin(theta));
}

function eciToThree(eciPos) {
  const scale = 1 / 6371; 
  return new THREE.Vector3(eciPos.x * scale, eciPos.z * scale, -eciPos.y * scale);
}

async function initEarth(container) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
  camera.position.set(0, 1.5, 5.0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);

  // --- LIGHTING & LENS FLARE ---
  scene.add(new THREE.AmbientLight(0x111111, 1.0)); 
  const sunLight = new THREE.DirectionalLight(0xffffff, 2.5); 
  
  // Position the sun based on real UTC time
  const now = new Date();
  const hours = now.getUTCHours() + now.getUTCMinutes() / 60;
  const angle = (hours / 24) * Math.PI * 2;
  sunLight.position.set(Math.cos(angle) * 10, 2, Math.sin(angle) * 10);
  scene.add(sunLight);

  // Add Lens Flare to Sun
  const textureLoader = new THREE.TextureLoader();
  textureLoader.crossOrigin = 'anonymous';
  const flareTexture = textureLoader.load(TEX.flare);
  const flareTexture2 = textureLoader.load(TEX.flare2);
  
  const lensflare = new Lensflare();
  lensflare.addElement(new LensflareElement(flareTexture, 400, 0));
  lensflare.addElement(new LensflareElement(flareTexture2, 60, 0.6));
  lensflare.addElement(new LensflareElement(flareTexture2, 70, 0.7));
  sunLight.add(lensflare);

  // --- TEXTURES ---
  const loader = new THREE.TextureLoader();
  loader.crossOrigin = 'anonymous';
  
  let map, normalMap, specularMap, cloudMap;
  try {
    [map, normalMap, specularMap, cloudMap] = await Promise.all([
      loader.loadAsync(TEX.diffuse), loader.loadAsync(TEX.normal), loader.loadAsync(TEX.specular), loader.loadAsync(TEX.clouds)
    ]);
    const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
    [map, cloudMap].forEach(t => { if(t) t.anisotropy = maxAnisotropy; });
    map.colorSpace = THREE.SRGBColorSpace; 
    cloudMap.colorSpace = THREE.SRGBColorSpace;
  } catch (err) { console.warn("Texture load error", err); }

  const earthGroup = new THREE.Group();
  scene.add(earthGroup);

  // --- EARTH & CLOUDS ---
  const earthMat = new THREE.MeshStandardMaterial({ 
    map, normalMap, roughnessMap: specularMap, metalness: 0.1, roughness: 0.9 
  });
  const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 128), earthMat);
  earthGroup.add(earth);

  if (cloudMap) {
    earthGroup.add(new THREE.Mesh(new THREE.SphereGeometry(1.005, 128, 128), new THREE.MeshStandardMaterial({ map: cloudMap, transparent: true, opacity: 0.5, depthWrite: false })));
  }

  // --- ATMOSPHERE ---
  const atmosphereMat = new THREE.ShaderMaterial({
    vertexShader: `varying vec3 vNormal; varying vec3 vPos; void main() { vNormal = normalize(normalMatrix * normal); vec4 mv = modelViewMatrix * vec4(position, 1.0); vPos = mv.xyz; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vNormal; varying vec3 vPos; void main() { vec3 viewDir = normalize(-vPos); float fres = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 3.0); gl_FragColor = vec4(vec3(0.2, 0.6, 1.0), fres * 0.8); }`,
    transparent: true, blending: THREE.AdditiveBlending, side: THREE.BackSide, depthWrite: false
  });
  earthGroup.add(new THREE.Mesh(new THREE.SphereGeometry(1.04, 64, 64), atmosphereMat));

  // --- STARS ---
  const starGeo = new THREE.BufferGeometry();
  const starPos = new Float32Array(5000 * 3);
  for (let i = 0; i < 5000; i++) {
    const r = 20 + Math.random() * 50; const th = Math.random() * Math.PI * 2; const ph = Math.acos(2 * Math.random() - 1);
    starPos[i*3] = r * Math.sin(ph) * Math.cos(th); starPos[i*3+1] = r * Math.cos(ph); starPos[i*3+2] = r * Math.sin(ph) * Math.sin(th);
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.1, transparent: true, opacity: 0.6 })));

  // --- DEBRIS ---
  const debGeo = new THREE.OctahedronGeometry(0.01);
  const debMat = new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.9, roughness: 0.3, flatShading: true });
  const debrisCount = 1500;
  const debrisMesh = new THREE.InstancedMesh(debGeo, debMat, debrisCount);
  const dummy = new THREE.Object3D();
  const debData = [];
  
  for (let i = 0; i < debrisCount; i++) {
    debData.push({
      r: 1.05 + Math.random() * 0.6, th: Math.random() * Math.PI * 2, ph: Math.acos(2 * Math.random() - 1),
      rot: new THREE.Vector3(Math.random(), Math.random(), Math.random()), speed: 0.05 + Math.random() * 0.1
    });
  }
  earthGroup.add(debrisMesh);

  // --- SATELLITES & FULL ORBITAL ELLIPSES ---
  const satModels = [];
  const orbitalLines = [];
  let satrecs = [];
  if (window.satellite) satrecs = TLES.map(t => window.satellite.twoline2satrec(t.line1, t.line2));

  function createSatellite() {
    const group = new THREE.Group();
    group.add(new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.05, 0.04), new THREE.MeshStandardMaterial({ color: 0xffaa00, metalness: 1.0, roughness: 0.2 })));
    const panelMat = new THREE.MeshStandardMaterial({ color: 0x0a1a3a, metalness: 0.9, roughness: 0.1, emissive: 0x06b6d4, emissiveIntensity: 0.3 });
    const p1 = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.002, 0.08), panelMat); p1.position.set(-0.095, 0, 0);
    const p2 = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.002, 0.08), panelMat); p2.position.set(0.095, 0, 0);
    group.add(p1, p2);
    return group;
  }

  // Generate the Entire Orbital Path
  for (let i = 0; i < TLES.length; i++) {
    const sat = createSatellite();
    earthGroup.add(sat);
    satModels.push(sat);

    if (window.satellite) {
      const satrec = satrecs[i];
      const points = [];
      // Calculate 100 points over the orbital period
      for (let t = 0; t < 100; t++) {
        const timeMs = now.getTime() + (t / 100) * TLES[i].period * 60 * 1000;
        try {
          const posVel = window.satellite.propagate(satrec, new Date(timeMs));
          if (posVel && posVel.position) {
             const vec = eciToThree(posVel.position);
             // Compensate for Earth's rotation for the static orbit line
             vec.applyMatrix4(new THREE.Matrix4().makeRotationY(-earth.rotation.y)); // Note: This might need adjustment as earth rotates
             points.push(vec.x, vec.y, vec.z);
          }
        } catch(e){}
      }

      if (points.length > 3) {
        const geometry = new LineGeometry();
        geometry.setPositions(points);
        const material = new LineMaterial({
            color: 0x22d3ee,
            linewidth: 1.5,
            transparent: true,
            opacity: 0.2,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        material.resolution.set(container.clientWidth, container.clientHeight);
        const line = new Line2(geometry, material);
        earthGroup.add(line);
        orbitalLines.push(line);
      }
    }
  }

  // --- CLIMATE DATA ---
  let climateMesh = null;
  window.addEventListener('climate-data-loaded', (e) => {
    if (climateMesh) earthGroup.remove(climateMesh);
    const positions = []; const colors = [];
    e.detail.forEach(obs => {
      const vec = latLonToVec3(obs.lat, obs.lon, 1.015);
      positions.push(vec.x, vec.y, vec.z);
      const color = new THREE.Color(obs.flag ? 0xef4444 : 0x22c55e);
      colors.push(color.r, color.g, color.b);
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    climateMesh = new THREE.Points(geo, new THREE.PointsMaterial({ 
      size: 0.025, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false 
    }));
    climateMesh.userData = e.detail;
    earthGroup.add(climateMesh);
  });

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.minDistance = 1.5; controls.maxDistance = 15; controls.autoRotate = true; controls.autoRotateSpeed = 0.2;

  const raycaster = new THREE.Raycaster(); raycaster.params.Points.threshold = 0.03;
  renderer.domElement.addEventListener('pointerdown', (ev) => {
    if (!climateMesh) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(climateMesh);
    if (hits.length > 0) {
      window.dispatchEvent(new CustomEvent('earth-obs-select', { detail: climateMesh.userData[hits[0].index] }));
      controls.autoRotate = false;
    }
  });

  window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
    orbitalLines.forEach(l => l.material.resolution.set(w, h));
  });

  const clock = new THREE.Clock();
  (function animate() {
    const dt = clock.getDelta();
    const nowTime = new Date();
    
    earth.rotation.y += dt * 0.05;

    // Update Debris
    for (let i = 0; i < debrisCount; i++) {
      const d = debData[i]; d.th += dt * d.speed;
      dummy.position.set(d.r * Math.sin(d.ph) * Math.cos(d.th), d.r * Math.cos(d.ph), d.r * Math.sin(d.ph) * Math.sin(d.th));
      dummy.rotation.set(d.rot.x + d.th, d.rot.y + d.th, d.rot.z + d.th);
      dummy.updateMatrix();
      debrisMesh.setMatrixAt(i, dummy.matrix);
    }
    debrisMesh.instanceMatrix.needsUpdate = true;

    // Update Satellites
    if (window.satellite) {
      for (let i = 0; i < satrecs.length; i++) {
        try {
          const posVel = window.satellite.propagate(satrecs[i], nowTime);
          if (posVel && posVel.position) {
            const threePos = eciToThree(posVel.position);
            threePos.applyMatrix4(new THREE.Matrix4().makeRotationY(-earth.rotation.y));
            satModels[i].position.copy(threePos);
            satModels[i].lookAt(earthGroup.position);
          }
        } catch (e) {}
      }
    }

    controls.update(); 
    renderer.render(scene, camera); 
    requestAnimationFrame(animate);
  })();
}
