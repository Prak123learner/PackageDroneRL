/* ═══════════════════════════════════════════════════════════════════════════
   renderer.js – Pure-Canvas 3D drone world renderer (no Three.js needed)
   ═══════════════════════════════════════════════════════════════════════════
   Renders the 200×200×200 cubic world with:
     • 3D perspective projection with orbit controls (mouse drag)
     • Ground grid, axis indicators
     • AABB box obstacles (shaded 3D boxes)
     • Drone (animated quadcopter icon) with flight-phase visuals
     • Target beacon (pulsing glow)
     • Flight trail (colour-coded by phase)
     • A* path corridor (dashed line)
     • Nearby-obstacle sensor ring
     • Cruise altitude plane indicator
   ═══════════════════════════════════════════════════════════════════════════ */

class DroneRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // Camera
    this.camYaw   = -0.6;  // radians
    this.camPitch = 0.45;
    this.camDist  = 400;
    this.camTarget = { x: 100, y: 100, z: 15 };
    this.viewMode = 'perspective'; // 'perspective' | 'top' | 'side'

    // World data
    this.worldSize = 200;
    this.dronePos = null;
    this.droneVel = null;
    this.targetPos = null;
    this.obstacles = [];       // full list {position, size_x, size_y, size_z, obstacle_type}
    this.nearbyObs = [];       // nearby [{relative_x, ...}]
    this.waypoint = null;
    this.trail = [];
    this.maxTrail = 500;
    this.flightPhase = 'GROUND';
    this.cruiseAltitude = 15;

    // Animation
    this._frame = 0;
    this._animId = null;
    this._dragging = false;
    this._lastMouse = { x: 0, y: 0 };

    this._onResize = this._resize.bind(this);
    this._onMouseDown = this._mouseDown.bind(this);
    this._onMouseMove = this._mouseMove.bind(this);
    this._onMouseUp   = this._mouseUp.bind(this);
    this._onWheel     = this._wheel.bind(this);

    window.addEventListener('resize', this._onResize);
    canvas.addEventListener('mousedown', this._onMouseDown);
    window.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('mouseup',   this._onMouseUp);
    canvas.addEventListener('wheel', this._onWheel, { passive: false });

    this._resize();
    this._animate();
  }

  /* ── Public API ─────────────────────────────────────────────────────── */

  updateDrone(pos, vel) {
    this.dronePos = pos;
    this.droneVel = vel;
    if (pos) {
      this.trail.push({ x: pos.x, y: pos.y, z: pos.z, phase: this.flightPhase });
      if (this.trail.length > this.maxTrail) this.trail.shift();
    }
  }

  updateTarget(pos) { this.targetPos = pos; }
  updateObstacles(list) { this.obstacles = list || []; }
  updateNearby(list) { this.nearbyObs = list || []; }
  updateWaypoint(wp) { this.waypoint = wp; }
  clearTrail() { this.trail = []; }

  updateFlightPhase(phase) { this.flightPhase = phase || 'GROUND'; }
  updateCruiseAltitude(alt) { this.cruiseAltitude = alt || 15; }

  setView(mode) {
    this.viewMode = mode;
    if (mode === 'top')  { this.camPitch = Math.PI / 2 - 0.01; this.camYaw = 0; }
    if (mode === 'side') { this.camPitch = 0.05; this.camYaw = 0; }
    if (mode === 'perspective') { this.camPitch = 0.45; this.camYaw = -0.6; }
  }

  destroy() {
    cancelAnimationFrame(this._animId);
    window.removeEventListener('resize', this._onResize);
    this.canvas.removeEventListener('mousedown', this._onMouseDown);
    window.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mouseup', this._onMouseUp);
    this.canvas.removeEventListener('wheel', this._onWheel);
  }

  /* ── Camera Math ────────────────────────────────────────────────────── */

  _project(x, y, z) {
    // Centre world
    const wx = x - this.camTarget.x;
    const wy = y - this.camTarget.y;
    const wz = z - this.camTarget.z;

    const cosY = Math.cos(this.camYaw), sinY = Math.sin(this.camYaw);
    const cosP = Math.cos(this.camPitch), sinP = Math.sin(this.camPitch);

    // Rotate yaw (around Z-up)
    const rx =  cosY * wx + sinY * wy;
    const ry = -sinY * wx + cosY * wy;
    const rz =  wz;

    // Rotate pitch
    const px = rx;
    const py = cosP * ry - sinP * rz;
    const pz = sinP * ry + cosP * rz;

    // Perspective
    const d = this.camDist + py;
    if (d < 1) return null;
    const fov = 600;
    const scale = fov / d;
    const sx = this.canvas.width  / 2 + px * scale;
    const sy = this.canvas.height / 2 - pz * scale;

    return { x: sx, y: sy, scale, depth: d };
  }

  /* ── Render loop ────────────────────────────────────────────────────── */

  _animate() {
    this._frame++;
    this._draw();
    this._animId = requestAnimationFrame(() => this._animate());
  }

  _draw() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    const bg = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W*0.7);
    bg.addColorStop(0, '#111320');
    bg.addColorStop(1, '#0a0b11');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Draw order (back-to-front approximation)
    this._drawGrid(ctx);
    this._drawAxes(ctx);
    this._drawWorldBounds(ctx);
    this._drawCruiseAltPlane(ctx);
    this._drawTrail(ctx);
    this._drawWaypoint(ctx);
    this._drawObstacles(ctx);
    this._drawTarget(ctx);
    this._drawDrone(ctx);
    this._drawSensorRing(ctx);
    this._drawCompass(ctx, W, H);
  }

  _drawGrid(ctx) {
    const s = this.worldSize;
    const step = 20;  // larger step for 200m world
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= s; i += step) {
      // X lines
      const a = this._project(i, 0, 0);
      const b = this._project(i, s, 0);
      if (a && b) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      // Y lines
      const c = this._project(0, i, 0);
      const d = this._project(s, i, 0);
      if (c && d) { ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke(); }
    }
  }

  _drawAxes(ctx) {
    const axLen = 20;  // longer axes for larger world
    const o = this._project(0, 0, 0);
    const ax = this._project(axLen, 0, 0);
    const ay = this._project(0, axLen, 0);
    const az = this._project(0, 0, axLen);
    if (!o) return;
    const draw = (to, color, label) => {
      if (!to) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(to.x, to.y); ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = '600 11px Inter, sans-serif';
      ctx.fillText(label, to.x + 4, to.y - 4);
    };
    draw(ax, '#fb7185', 'X');
    draw(ay, '#34d399', 'Y');
    draw(az, '#818cf8', 'Z');
  }

  _drawWorldBounds(ctx) {
    const s = this.worldSize;
    const edges = [
      [[0,0,0],[s,0,0]], [[s,0,0],[s,s,0]], [[s,s,0],[0,s,0]], [[0,s,0],[0,0,0]],
      [[0,0,s],[s,0,s]], [[s,0,s],[s,s,s]], [[s,s,s],[0,s,s]], [[0,s,s],[0,0,s]],
      [[0,0,0],[0,0,s]], [[s,0,0],[s,0,s]], [[s,s,0],[s,s,s]], [[0,s,0],[0,s,s]],
    ];
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.08)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    for (const [a, b] of edges) {
      const pa = this._project(...a);
      const pb = this._project(...b);
      if (pa && pb) {
        ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }

  _drawCruiseAltPlane(ctx) {
    // Draw a subtle translucent plane at cruise altitude
    const s = this.worldSize;
    const z = this.cruiseAltitude;
    const corners = [
      this._project(0, 0, z),
      this._project(s, 0, z),
      this._project(s, s, z),
      this._project(0, s, z),
    ].filter(Boolean);

    if (corners.length === 4) {
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
      ctx.closePath();
      ctx.fillStyle = 'rgba(6, 182, 212, 0.03)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.1)';
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 8]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label
      const labelPos = this._project(s + 2, s / 2, z);
      if (labelPos) {
        ctx.fillStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.font = '500 9px Inter, sans-serif';
        ctx.fillText(`Cruise: ${z.toFixed(0)}m`, labelPos.x, labelPos.y);
      }
    }
  }

  _drawTrail(ctx) {
    if (this.trail.length < 2) return;
    ctx.lineWidth = 2;

    const phaseColors = {
      GROUND:     '244, 63, 94',    // rose
      LIFTING:    '99, 102, 241',   // indigo
      CRUISING:   '6, 182, 212',    // cyan
      DESCENDING: '245, 158, 11',   // amber
      LANDED:     '16, 185, 129',   // emerald
    };

    for (let i = 1; i < this.trail.length; i++) {
      const t = this.trail[i];
      const p = this._project(t.x, t.y, t.z);
      const pp = this._project(this.trail[i-1].x, this.trail[i-1].y, this.trail[i-1].z);
      if (!p || !pp) continue;
      const alpha = (i / this.trail.length) * 0.7;
      const rgb = phaseColors[t.phase] || '99, 102, 241';
      ctx.strokeStyle = `rgba(${rgb}, ${alpha})`;
      ctx.beginPath(); ctx.moveTo(pp.x, pp.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    }
  }

  _drawWaypoint(ctx) {
    if (!this.waypoint) return;
    const p = this._project(this.waypoint.x, this.waypoint.y, this.waypoint.z);
    if (!p) return;
    const pulse = 0.5 + 0.5 * Math.sin(this._frame * 0.08);
    const r = 5 + pulse * 3;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * p.scale / 5, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(6, 182, 212, ${0.3 + pulse * 0.3})`;
    ctx.fill();
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  _drawObstacles(ctx) {
    // Sort by depth (furthest first)
    const projected = this.obstacles.map(obs => {
      const p = this._project(obs.position.x, obs.position.y, obs.position.z);
      return { obs, p };
    }).filter(o => o.p).sort((a, b) => b.p.depth - a.p.depth);

    for (const { obs, p } of projected) {
      const isNearby = this.nearbyObs.some(n => n.id === obs.id);

      // Get box dimensions (support both old radius-based and new AABB)
      const sx = obs.size_x || (obs.radius ? obs.radius * 2 : 2);
      const sy = obs.size_y || (obs.radius ? obs.radius * 2 : 2);
      const sz = obs.size_z || (obs.radius ? obs.radius * 2 : 10);

      // Convert 3D box to screen — draw the visible face as a rectangle
      // We'll draw 3 visible faces of the box for a 3D look
      const cx = obs.position.x;
      const cy = obs.position.y;
      const cz = obs.position.z;
      const hx = sx / 2, hy = sy / 2, hz = sz / 2;

      // 8 corners of the AABB
      const corners3D = [
        [cx - hx, cy - hy, cz - hz], // 0: front-bottom-left
        [cx + hx, cy - hy, cz - hz], // 1: front-bottom-right
        [cx + hx, cy + hy, cz - hz], // 2: back-bottom-right
        [cx - hx, cy + hy, cz - hz], // 3: back-bottom-left
        [cx - hx, cy - hy, cz + hz], // 4: front-top-left
        [cx + hx, cy - hy, cz + hz], // 5: front-top-right
        [cx + hx, cy + hy, cz + hz], // 6: back-top-right
        [cx - hx, cy + hy, cz + hz], // 7: back-top-left
      ];

      const corners = corners3D.map(c => this._project(c[0], c[1], c[2]));
      if (corners.some(c => !c)) continue;

      // Determine face colors based on type and nearby status
      const typeColors = {
        building: { base: 'rgba(120, 113, 170', edge: 'rgba(80, 75, 120' },
        tower:    { base: 'rgba(100, 140, 180', edge: 'rgba(60, 90, 120' },
        tree:     { base: 'rgba(80, 160, 100',  edge: 'rgba(50, 110, 60' },
        antenna:  { base: 'rgba(180, 140, 80',  edge: 'rgba(120, 90, 50' },
      };
      const colors = typeColors[obs.obstacle_type] || typeColors.building;

      const baseFill = isNearby ? 'rgba(244, 63, 94' : colors.base;
      const edgeFill = isNearby ? 'rgba(200, 40, 70' : colors.edge;
      const baseAlpha = isNearby ? 0.6 : 0.4;
      const edgeAlpha = isNearby ? 0.7 : 0.5;

      // Draw 3 faces: front, right side, top — using painter's algorithm approximation
      // Front face (0,1,5,4)
      this._drawFace(ctx, [corners[0], corners[1], corners[5], corners[4]],
        `${edgeFill}, ${edgeAlpha})`, isNearby);
      // Right face (1,2,6,5)
      this._drawFace(ctx, [corners[1], corners[2], corners[6], corners[5]],
        `${baseFill}, ${baseAlpha * 0.8})`, isNearby);
      // Top face (4,5,6,7)
      this._drawFace(ctx, [corners[4], corners[5], corners[6], corners[7]],
        `${baseFill}, ${baseAlpha})`, isNearby);

      // Outline edges
      ctx.strokeStyle = isNearby ? 'rgba(244, 63, 94, 0.6)' : 'rgba(99, 102, 241, 0.2)';
      ctx.lineWidth = isNearby ? 1.5 : 0.8;
      // Draw all 12 edges
      const edgeIndices = [
        [0,1],[1,2],[2,3],[3,0], // bottom
        [4,5],[5,6],[6,7],[7,4], // top
        [0,4],[1,5],[2,6],[3,7], // verticals
      ];
      for (const [a, b] of edgeIndices) {
        if (corners[a] && corners[b]) {
          ctx.beginPath();
          ctx.moveTo(corners[a].x, corners[a].y);
          ctx.lineTo(corners[b].x, corners[b].y);
          ctx.stroke();
        }
      }

      // Label
      const topCenter = corners[6]; // use a top corner
      if (topCenter && p.scale > 0.3) {
        ctx.fillStyle = 'rgba(255,255,255,0.45)';
        const fontSize = Math.max(7, Math.min(11, 10 * p.scale / 2));
        ctx.font = `500 ${fontSize}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(obs.obstacle_type, topCenter.x, topCenter.y - 6);
        ctx.textAlign = 'start';
      }
    }
  }

  _drawFace(ctx, corners, fillStyle, highlighted) {
    if (corners.some(c => !c)) return;
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < corners.length; i++) {
      ctx.lineTo(corners[i].x, corners[i].y);
    }
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }

  _drawTarget(ctx) {
    if (!this.targetPos) return;
    const p = this._project(this.targetPos.x, this.targetPos.y, this.targetPos.z);
    if (!p) return;

    const pulse = 0.5 + 0.5 * Math.sin(this._frame * 0.06);
    const r = 12 * p.scale / 5;

    // Glow
    const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 3);
    glow.addColorStop(0, `rgba(16, 185, 129, ${0.25 + pulse * 0.15})`);
    glow.addColorStop(1, 'rgba(16, 185, 129, 0)');
    ctx.fillStyle = glow;
    ctx.fillRect(p.x - r * 3, p.y - r * 3, r * 6, r * 6);

    // Landing pad circle
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * (0.8 + pulse * 0.2), 0, Math.PI * 2);
    ctx.fillStyle = `rgba(16, 185, 129, ${0.5 + pulse * 0.3})`;
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.stroke();

    // H landing pad marker
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
    ctx.lineWidth = 2;
    const hr = r * 0.5;
    // Draw "H"
    ctx.beginPath();
    ctx.moveTo(p.x - hr, p.y - hr); ctx.lineTo(p.x - hr, p.y + hr);
    ctx.moveTo(p.x - hr, p.y);      ctx.lineTo(p.x + hr, p.y);
    ctx.moveTo(p.x + hr, p.y - hr); ctx.lineTo(p.x + hr, p.y + hr);
    ctx.stroke();

    // Crosshair
    const cr = r * 1.5;
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(p.x - cr, p.y); ctx.lineTo(p.x + cr, p.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.x, p.y - cr); ctx.lineTo(p.x, p.y + cr); ctx.stroke();

    // Vertical beam from ground to sky (landing guide)
    const groundP = this._project(this.targetPos.x, this.targetPos.y, 0);
    const skyP = this._project(this.targetPos.x, this.targetPos.y, this.cruiseAltitude);
    if (groundP && skyP) {
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.08)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 8]);
      ctx.beginPath(); ctx.moveTo(groundP.x, groundP.y); ctx.lineTo(skyP.x, skyP.y); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Label
    ctx.fillStyle = '#34d399';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('LANDING', p.x, p.y + r * 2.5);
    ctx.textAlign = 'start';
  }

  _drawDrone(ctx) {
    if (!this.dronePos) return;
    const p = this._project(this.dronePos.x, this.dronePos.y, this.dronePos.z);
    if (!p) return;

    const s = Math.max(6, 14 * p.scale / 5);
    const spin = this._frame * 0.15;

    // Phase-based glow color
    const phaseGlow = {
      GROUND:     { r: 244, g: 63,  b: 94 },   // rose
      LIFTING:    { r: 139, g: 92,  b: 246 },   // violet
      CRUISING:   { r: 6,   g: 182, b: 212 },   // cyan
      DESCENDING: { r: 245, g: 158, b: 11 },    // amber
      LANDED:     { r: 16,  g: 185, b: 129 },   // emerald
    };
    const gc = phaseGlow[this.flightPhase] || phaseGlow.GROUND;

    // Shadow on ground
    const gp = this._project(this.dronePos.x, this.dronePos.y, 0);
    if (gp) {
      ctx.beginPath();
      ctx.ellipse(gp.x, gp.y, s * 0.8, s * 0.3, 0, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.2)';
      ctx.fill();
      // Altitude line
      ctx.strokeStyle = `rgba(${gc.r}, ${gc.g}, ${gc.b}, 0.15)`;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 5]);
      ctx.beginPath(); ctx.moveTo(gp.x, gp.y); ctx.lineTo(p.x, p.y); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Phase-colored glow
    const glw = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, s * 2.5);
    glw.addColorStop(0, `rgba(${gc.r}, ${gc.g}, ${gc.b}, 0.3)`);
    glw.addColorStop(1, `rgba(${gc.r}, ${gc.g}, ${gc.b}, 0)`);
    ctx.fillStyle = glw;
    ctx.fillRect(p.x - s * 2.5, p.y - s * 2.5, s * 5, s * 5);

    // Body
    ctx.beginPath();
    ctx.arc(p.x, p.y, s * 0.45, 0, Math.PI * 2);
    ctx.fillStyle = '#6366f1';
    ctx.fill();
    ctx.strokeStyle = '#818cf8';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Arms & rotors (spin faster during lifting)
    const rotorSpeed = this.flightPhase === 'LIFTING' ? 0.25 :
                       this.flightPhase === 'GROUND'  ? 0.05 :
                       this.flightPhase === 'LANDED'  ? 0.02 : 0.15;
    const actualSpin = this._frame * rotorSpeed;

    for (let i = 0; i < 4; i++) {
      const a = actualSpin + (Math.PI / 2) * i;
      const ex = p.x + Math.cos(a) * s;
      const ey = p.y + Math.sin(a) * s;

      // Arm
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(ex, ey); ctx.stroke();

      // Rotor disc
      ctx.beginPath();
      ctx.arc(ex, ey, s * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${gc.r}, ${gc.g}, ${gc.b}, 0.2)`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${gc.r}, ${gc.g}, ${gc.b}, 0.5)`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Center highlight
    ctx.beginPath();
    ctx.arc(p.x - s * 0.1, p.y - s * 0.1, s * 0.15, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fill();

    // Label with phase
    ctx.fillStyle = '#c7d2fe';
    ctx.font = '700 10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('DRONE', p.x, p.y - s - 6);
    ctx.textAlign = 'start';
  }

  _drawSensorRing(ctx) {
    if (!this.dronePos) return;
    const sensorRange = 25; // matches SENSOR_RANGE
    // Draw a subtle circle at the drone's altitude
    const segments = 36;
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.08)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      const px = this.dronePos.x + Math.cos(a) * sensorRange;
      const py = this.dronePos.y + Math.sin(a) * sensorRange;
      const p = this._project(px, py, this.dronePos.z);
      if (!p) continue;
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _drawCompass(ctx, W, H) {
    const cx = 50, cy = H - 50, r = 25;
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

    const dirs = [
      { label: 'N', angle: -Math.PI / 2 },
      { label: 'E', angle: 0 },
      { label: 'S', angle: Math.PI / 2 },
      { label: 'W', angle: Math.PI },
    ];
    ctx.font = '600 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const d of dirs) {
      const a = d.angle + this.camYaw;
      const tx = cx + Math.cos(a) * (r + 10);
      const ty = cy + Math.sin(a) * (r + 10);
      ctx.fillStyle = d.label === 'N' ? '#fb7185' : 'rgba(176, 181, 201, 0.5)';
      ctx.fillText(d.label, tx, ty);
    }
    ctx.textBaseline = 'alphabetic';
  }

  /* ── Interaction ────────────────────────────────────────────────────── */

  _mouseDown(e) {
    this._dragging = true;
    this._lastMouse = { x: e.clientX, y: e.clientY };
    this.viewMode = 'perspective';
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('[data-view="perspective"]')?.classList.add('active');
  }

  _mouseMove(e) {
    if (!this._dragging) return;
    const dx = e.clientX - this._lastMouse.x;
    const dy = e.clientY - this._lastMouse.y;
    this.camYaw   -= dx * 0.005;
    this.camPitch += dy * 0.005;
    this.camPitch = Math.max(-0.1, Math.min(Math.PI / 2 - 0.01, this.camPitch));
    this._lastMouse = { x: e.clientX, y: e.clientY };
  }

  _mouseUp() { this._dragging = false; }

  _wheel(e) {
    e.preventDefault();
    this.camDist += e.deltaY * 0.3;
    this.camDist = Math.max(50, Math.min(800, this.camDist));
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width  = rect.width;
    this.canvas.height = rect.height;
  }
}

// Expose globally
window.DroneRenderer = DroneRenderer;
