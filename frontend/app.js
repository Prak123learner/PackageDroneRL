/* ═══════════════════════════════════════════════════════════════════════════
   app.js – Application logic for the Drone Delivery Simulator frontend
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
  'use strict';

  // ────────────────────────────────────────────────────────────────────────
  //  DOM references
  // ────────────────────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  const apiUrlInput   = $('api-url');
  const connBadge     = $('connection-badge');

  // Locations
  const startX = $('start-x'), startY = $('start-y'), startZ = $('start-z');
  const endX   = $('end-x'),   endY   = $('end-y'),   endZ   = $('end-z');
  const numObsInput   = $('num-obstacles');
  const seedInput     = $('seed');

  // Acceleration sliders
  const axSlider = $('ax'), aySlider = $('ay'), azSlider = $('az');
  const axVal = $('ax-val'), ayVal = $('ay-val'), azVal = $('az-val');


  // Buttons
  const btnReset = $('btn-reset');
  const btnStep  = $('btn-step');
  const btnAuto  = $('btn-auto');
  const btnStop  = $('btn-stop');

  // Sim speed
  const simSpeedSlider = $('sim-speed');
  const simSpeedVal    = $('sim-speed-val');

  // Overlay
  const overlayEpisode = $('overlay-episode');
  const overlayStep    = $('overlay-step');
  const overlayReward  = $('overlay-reward');
  const overlayDist    = $('overlay-dist');
  const overlayAlt     = $('overlay-alt');
  const overlayPhase   = $('overlay-phase');
  const overlayStatus  = $('overlay-status');

  // Telemetry
  const telemPos       = $('telem-pos');
  const telemVel       = $('telem-vel');
  const telemAccel     = $('telem-accel');
  const telemSpeed     = $('telem-speed');
  const telemAlt       = $('telem-alt');
  const telemCruiseAlt = $('telem-cruise-alt');
  const telemTarget    = $('telem-target');
  const telemDist      = $('telem-dist');
  const telemHDist     = $('telem-hdist');
  const telemDir       = $('telem-dir');
  const telemWP        = $('telem-wp');
  const telemPath      = $('telem-path');

  // Status
  const flagDelivered = $('flag-delivered');
  const flagCollision = $('flag-collision');
  const flagOOB       = $('flag-oob');
  const flagDone      = $('flag-done');
  const telemFlightPhase = $('telem-flight-phase');
  const telemStepReward  = $('telem-step-reward');
  const telemTotalReward = $('telem-total-reward');
  const telemStepsLeft   = $('telem-steps-left');
  const telemObsCount    = $('telem-obs-count');

  // Obstacles list
  const obsList = $('obstacles-list');

  // Reward chart
  const rewardCanvas = $('reward-chart');

  // Flight phase steps
  const phaseSteps = ['GROUND', 'LIFTING', 'CRUISING', 'DESCENDING', 'LANDED'];
  const phaseElements = {
    GROUND:     $('phase-ground'),
    LIFTING:    $('phase-lifting'),
    CRUISING:   $('phase-cruising'),
    DESCENDING: $('phase-descending'),
    LANDED:     $('phase-landed'),
  };

  // ────────────────────────────────────────────────────────────────────────
  //  State
  // ────────────────────────────────────────────────────────────────────────
  let renderer = null;
  let autoInterval = null;
  let isRunning = false;
  let currentObs = null;         // latest DroneObservation from API
  let allObstacles = [];         // full obstacle list (from /obstacles)
  let customObstacles = [];      // user-defined obstacles for reset
  let rewardHistory = [];
  let totalRewardAcc = 0;

  // ────────────────────────────────────────────────────────────────────────
  //  Helpers
  // ────────────────────────────────────────────────────────────────────────
  function apiBase() { return apiUrlInput.value.replace(/\/+$/, ''); }

  function toast(msg, type = 'info', duration = 3000) {
    const area = $('toast-area');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    area.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 400); }, duration);
  }

  function fmt(v, d = 2) {
    if (v == null) return '—';
    return Number(v).toFixed(d);
  }

  async function api(method, path, body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(apiBase() + path, opts);
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`API ${res.status}: ${err}`);
    }
    return res.json();
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Connection check
  // ────────────────────────────────────────────────────────────────────────
  async function checkConnection() {
    try {
      await api('GET', '/health');
      connBadge.classList.add('connected');
      connBadge.querySelector('.label').textContent = 'Connected';
      return true;
    } catch {
      connBadge.classList.remove('connected');
      connBadge.querySelector('.label').textContent = 'Disconnected';
      return false;
    }
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Update flight phase indicator
  // ────────────────────────────────────────────────────────────────────────
  function updateFlightPhaseUI(phase) {
    const currentIdx = phaseSteps.indexOf(phase);

    for (const [phaseName, el] of Object.entries(phaseElements)) {
      if (!el) continue;
      const idx = phaseSteps.indexOf(phaseName);
      el.classList.remove('active', 'completed');
      if (idx < currentIdx) {
        el.classList.add('completed');
      } else if (idx === currentIdx) {
        el.classList.add('active');
      }
    }

    // Update connectors
    const connectors = document.querySelectorAll('.phase-connector');
    connectors.forEach((conn, i) => {
      conn.classList.toggle('completed', i < currentIdx);
    });
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Update UI from observation
  // ────────────────────────────────────────────────────────────────────────
  function updateUI(obs) {
    currentObs = obs;

    // Renderer
    const pos = obs.position || {};
    const vel = obs.velocity || {};
    const accel = obs.acceleration || {};

    renderer.updateFlightPhase(obs.flight_phase || 'GROUND');
    renderer.updateCruiseAltitude(obs.cruise_altitude || 15);
    renderer.updateDrone(pos, vel);
    renderer.updateTarget(obs.target_position);
    renderer.updateNearby(obs.nearby_obstacles || []);
    renderer.updateWaypoint(obs.next_waypoint);

    // Overlay HUD
    overlayEpisode.textContent = `Episode: ${(obs.metadata?.episode_id || '—').slice(0, 8)}`;
    overlayStep.textContent = `Step: ${obs.metadata?.step || 0} / ${(obs.metadata?.step || 0) + (obs.steps_remaining ?? 2000)}`;
    overlayReward.textContent = `Reward: ${fmt(obs.metadata?.total_reward)}`;
    overlayDist.textContent = `Dist: ${fmt(obs.distance_to_target, 1)}m`;
    overlayAlt.textContent = `Alt: ${fmt(pos.z, 1)}m`;
    overlayPhase.textContent = `Phase: ${obs.flight_phase || 'GROUND'}`;

    // Phase-colored overlay
    const phaseColorClass = {
      GROUND: 'phase-ground-color',
      LIFTING: 'phase-lifting-color',
      CRUISING: 'phase-cruising-color',
      DESCENDING: 'phase-descending-color',
      LANDED: 'phase-landed-color',
    };
    overlayPhase.className = 'overlay-phase ' + (phaseColorClass[obs.flight_phase] || '');

    // Flight phase bar
    updateFlightPhaseUI(obs.flight_phase || 'GROUND');

    // Status overlay
    overlayStatus.classList.remove('show', 'delivered', 'collision', 'oob');
    if (obs.package_delivered) {
      overlayStatus.textContent = '📦 Package Delivered!';
      overlayStatus.classList.add('show', 'delivered');
    } else if (obs.collision_occurred) {
      overlayStatus.textContent = '💥 Collision!';
      overlayStatus.classList.add('show', 'collision');
    } else if (obs.out_of_bounds) {
      overlayStatus.textContent = '⚠ Out of Bounds';
      overlayStatus.classList.add('show', 'oob');
    }

    // Telemetry
    telemPos.textContent = `${fmt(pos.x)}, ${fmt(pos.y)}, ${fmt(pos.z)}`;
    telemVel.textContent = `${fmt(vel.vx)}, ${fmt(vel.vy)}, ${fmt(vel.vz)}`;
    telemAccel.textContent = `${fmt(accel.vx)}, ${fmt(accel.vy)}, ${fmt(accel.vz)}`;
    telemSpeed.textContent = `${fmt(obs.metadata?.speed)} m/s`;
    telemAlt.textContent = `${fmt(pos.z, 1)} m`;
    telemCruiseAlt.textContent = `${fmt(obs.cruise_altitude, 1)} m`;
    const tp = obs.target_position || {};
    telemTarget.textContent = `${fmt(tp.x)}, ${fmt(tp.y)}, ${fmt(tp.z)}`;
    telemDist.textContent = `${fmt(obs.distance_to_target, 1)} m`;
    telemHDist.textContent = `${fmt(obs.horizontal_distance_to_target, 1)} m`;
    const td = obs.target_direction || [0,0,0];
    telemDir.textContent = `${fmt(td[0])}, ${fmt(td[1])}, ${fmt(td[2])}`;
    const wp = obs.next_waypoint;
    telemWP.textContent = wp ? `${fmt(wp.x)}, ${fmt(wp.y)}, ${fmt(wp.z)}` : 'None';
    telemPath.textContent = `${obs.path_length || 0} waypoints`;

    // Flags
    flagDelivered.classList.toggle('active', !!obs.package_delivered);
    flagCollision.classList.toggle('active', !!obs.collision_occurred);
    flagOOB.classList.toggle('active', !!obs.out_of_bounds);
    flagDone.classList.toggle('active', !!obs.done);

    // Flight phase text
    telemFlightPhase.textContent = obs.flight_phase || '—';

    telemStepReward.textContent = fmt(obs.reward);
    telemTotalReward.textContent = fmt(obs.metadata?.total_reward);
    telemStepsLeft.textContent = obs.steps_remaining;
    telemObsCount.textContent = obs.metadata?.num_obstacles || '—';

    // Nearby obstacles panel — show box dimensions instead of radius
    const nearby = obs.nearby_obstacles || [];
    if (nearby.length === 0) {
      obsList.innerHTML = '<span class="muted">No obstacles in range</span>';
    } else {
      obsList.innerHTML = nearby.map(o => `
        <div class="obs-item">
          <span class="obs-type">${o.obstacle_type}</span>
          <span class="obs-dist">${fmt(o.distance, 1)}m (${fmt(o.size_x,0)}×${fmt(o.size_y,0)}×${fmt(o.size_z,0)})</span>
        </div>
      `).join('');
    }

    // Reward history
    rewardHistory.push(obs.reward || 0);
    if (rewardHistory.length > 200) rewardHistory.shift();
    drawRewardChart();
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Reward chart (mini spark-line)
  // ────────────────────────────────────────────────────────────────────────
  function drawRewardChart() {
    const ctx = rewardCanvas.getContext('2d');
    const W = rewardCanvas.parentElement.clientWidth - 32;
    const H = 100;
    rewardCanvas.width = W;
    rewardCanvas.height = H;
    ctx.clearRect(0, 0, W, H);

    if (rewardHistory.length < 2) return;

    const maxR = Math.max(1, ...rewardHistory.map(Math.abs));
    const mid = H / 2;

    // Zero line
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
    ctx.setLineDash([]);

    // Reward line
    ctx.beginPath();
    for (let i = 0; i < rewardHistory.length; i++) {
      const x = (i / (rewardHistory.length - 1)) * W;
      const y = mid - (rewardHistory[i] / maxR) * (mid - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(99, 102, 241, 0.15)');
    grad.addColorStop(0.5, 'rgba(99, 102, 241, 0)');
    grad.addColorStop(1, 'rgba(244, 63, 94, 0.1)');
    ctx.lineTo(W, mid);
    ctx.lineTo(0, mid);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Actions
  // ────────────────────────────────────────────────────────────────────────
  async function doReset() {
    try {
      const seedVal = seedInput.value.trim();
      const body = {
        num_obstacles: parseInt(numObsInput.value) || 15,
      };
      if (seedVal !== '') body.seed = parseInt(seedVal);

      // Send custom start / target positions
      const sx = parseFloat(startX.value), sy = parseFloat(startY.value);
      const ex = parseFloat(endX.value),   ey = parseFloat(endY.value);
      if (!isNaN(sx) && !isNaN(sy)) body.start_position = { x: sx, y: sy };
      if (!isNaN(ex) && !isNaN(ey)) body.target_position = { x: ex, y: ey };

      // Send custom obstacles if any
      if (customObstacles.length > 0) {
        body.obstacles = customObstacles.map(o => ({
          x: o.x, y: o.y, height: o.height,
          size_x: o.size_x || 2, size_y: o.size_y || 2,
          obstacle_type: o.obstacle_type || 'building',
        }));
      }

      const obs = await api('POST', '/reset', body);

      // Fetch full obstacle list for accurate 3D rendering
      try {
        const obsData = await api('GET', '/obstacles');
        allObstacles = obsData.obstacles || [];
        renderer.updateObstacles(allObstacles);
      } catch {
        allObstacles = [];
        renderer.updateObstacles([]);
      }

      renderer.clearTrail();
      rewardHistory = [];
      totalRewardAcc = 0;

      // Reset overlay status
      overlayStatus.classList.remove('show', 'delivered', 'collision', 'oob');

      updateUI(obs);
      const obsCount = customObstacles.length > 0
        ? `${customObstacles.length} custom`
        : `${body.num_obstacles} random`;
      toast(`🔄 Reset – ${obsCount} obstacles, start (${sx||'~'},${sy||'~'}) → target (${ex||'~'},${ey||'~'})`, 'success');
    } catch (e) {
      toast(`Reset failed: ${e.message}`, 'error');
    }
  }

  async function doStep() {
    try {
      const body = {
        ax: parseFloat(axSlider.value) || 0,
        ay: parseFloat(aySlider.value) || 0,
        az: parseFloat(azSlider.value) || 0,
      };
      const obs = await api('POST', '/step', body);

      // Track obstacles from nearby observations (AABB version)
      for (const n of (obs.nearby_obstacles || [])) {
        const dronePos = obs.position;
        const absPos = {
          x: dronePos.x + n.relative_x,
          y: dronePos.y + n.relative_y,
          z: dronePos.z + n.relative_z,
        };
        // Upsert
        const existing = allObstacles.find(o => o.id === n.id);
        if (existing) {
          existing.position = absPos;
          existing.size_x = n.size_x;
          existing.size_y = n.size_y;
          existing.size_z = n.size_z;
          existing.obstacle_type = n.obstacle_type;
        } else {
          allObstacles.push({
            id: n.id,
            position: absPos,
            size_x: n.size_x,
            size_y: n.size_y,
            size_z: n.size_z,
            obstacle_type: n.obstacle_type,
          });
        }
      }
      renderer.updateObstacles(allObstacles);

      updateUI(obs);

      if (obs.done && isRunning) {
        stopAutoRun();
        if (obs.package_delivered) toast('📦 Package delivered! Episode complete.', 'success');
        else if (obs.collision_occurred) toast('💥 Collision detected! Episode ended.', 'error');
        else if (obs.out_of_bounds) toast('⚠ Drone went out of bounds!', 'warn');
        else toast('⏱ Episode timed out.', 'warn');
      }
    } catch (e) {
      toast(`Step failed: ${e.message}`, 'error');
      if (isRunning) stopAutoRun();
    }
  }

  function startAutoRun() {
    if (isRunning) return;
    isRunning = true;
    btnAuto.disabled = true;
    btnStop.disabled = false;
    btnReset.disabled = true;
    const fps = parseInt(simSpeedSlider.value) || 10;
    autoInterval = setInterval(doStep, 1000 / fps);
    toast(`▶ Auto-run started at ${fps} steps/s`, 'info');
  }

  function stopAutoRun() {
    isRunning = false;
    clearInterval(autoInterval);
    autoInterval = null;
    btnAuto.disabled = false;
    btnStop.disabled = true;
    btnReset.disabled = false;
    toast('⏹ Auto-run stopped', 'info');
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Slider updates
  // ────────────────────────────────────────────────────────────────────────
  function setupSlider(slider, output) {
    const update = () => { output.textContent = parseFloat(slider.value).toFixed(1); };
    slider.addEventListener('input', update);
    update();
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Quick-preset chips
  // ────────────────────────────────────────────────────────────────────────
  function setupChips() {
    document.querySelectorAll('.chip[data-ax]').forEach(chip => {
      chip.addEventListener('click', () => {
        axSlider.value = chip.dataset.ax;
        aySlider.value = chip.dataset.ay;
        azSlider.value = chip.dataset.az;
        axVal.textContent = parseFloat(chip.dataset.ax).toFixed(1);
        ayVal.textContent = parseFloat(chip.dataset.ay).toFixed(1);
        azVal.textContent = parseFloat(chip.dataset.az).toFixed(1);
      });
    });
  }


  // ────────────────────────────────────────────────────────────────────────
  //  Custom obstacle editor
  // ────────────────────────────────────────────────────────────────────────
  function setupObstacleEditor() {
    const addBtn = $('btn-add-obstacle');
    const listEl = $('custom-obstacles-list');
    if (!addBtn || !listEl) return;

    addBtn.addEventListener('click', () => {
      const ox = parseFloat($('obs-x')?.value) || 100;
      const oy = parseFloat($('obs-y')?.value) || 100;
      const oh = parseFloat($('obs-h')?.value) || 15;
      const ot = $('obs-type')?.value || 'building';
      customObstacles.push({ x: ox, y: oy, height: oh, obstacle_type: ot });
      renderObstacleEditor();
    });
  }

  function renderObstacleEditor() {
    const listEl = $('custom-obstacles-list');
    if (!listEl) return;
    if (customObstacles.length === 0) {
      listEl.innerHTML = '<span class="muted">No custom obstacles – will use random</span>';
      return;
    }
    listEl.innerHTML = customObstacles.map((o, i) => `
      <div class="obs-item" data-idx="${i}">
        <span class="obs-type">${o.obstacle_type}</span>
        <span class="obs-dist">(${o.x}, ${o.y}) h=${o.height}m</span>
        <span class="obs-remove" data-idx="${i}" style="color:var(--rose);cursor:pointer;margin-left:auto">✕</span>
      </div>
    `).join('');
    listEl.querySelectorAll('.obs-remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        customObstacles.splice(parseInt(btn.dataset.idx), 1);
        renderObstacleEditor();
      });
    });
  }

  // ────────────────────────────────────────────────────────────────────────
  //  View mode buttons
  // ────────────────────────────────────────────────────────────────────────
  function setupViewButtons() {
    document.querySelectorAll('.view-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderer.setView(btn.dataset.view);
      });
    });
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Sim speed
  // ────────────────────────────────────────────────────────────────────────
  function setupSimSpeed() {
    const update = () => {
      const v = parseInt(simSpeedSlider.value);
      simSpeedVal.textContent = `${v} steps/s`;
      if (isRunning) {
        clearInterval(autoInterval);
        autoInterval = setInterval(doStep, 1000 / v);
      }
    };
    simSpeedSlider.addEventListener('input', update);
    update();
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Keyboard shortcuts
  // ────────────────────────────────────────────────────────────────────────
  function setupKeyboard() {
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      switch (e.key) {
        case 'r': doReset(); break;
        case ' ':
          e.preventDefault();
          if (isRunning) stopAutoRun();
          else doStep();
          break;
        case 'a': startAutoRun(); break;
        case 's': stopAutoRun(); break;
        case 'ArrowUp':
          azSlider.value = Math.min(5, parseFloat(azSlider.value) + 0.5);
          azVal.textContent = parseFloat(azSlider.value).toFixed(1);
          break;
        case 'ArrowDown':
          azSlider.value = Math.max(-5, parseFloat(azSlider.value) - 0.5);
          azVal.textContent = parseFloat(azSlider.value).toFixed(1);
          break;
        case 'ArrowRight':
          axSlider.value = Math.min(5, parseFloat(axSlider.value) + 0.5);
          axVal.textContent = parseFloat(axSlider.value).toFixed(1);
          break;
        case 'ArrowLeft':
          axSlider.value = Math.max(-5, parseFloat(axSlider.value) - 0.5);
          axVal.textContent = parseFloat(axSlider.value).toFixed(1);
          break;
      }
    });
  }

  // ────────────────────────────────────────────────────────────────────────
  //  Initialise
  // ────────────────────────────────────────────────────────────────────────
  function init() {
    // Create renderer
    const canvas = $('drone-canvas');
    renderer = new DroneRenderer(canvas);

    // Sliders
    setupSlider(axSlider, axVal);
    setupSlider(aySlider, ayVal);
    setupSlider(azSlider, azVal);

    // Chips
    setupChips();

    // Obstacle editor
    setupObstacleEditor();
    renderObstacleEditor();

    // View buttons
    setupViewButtons();

    // Sim speed
    setupSimSpeed();

    // Keyboard
    setupKeyboard();

    // Button handlers
    btnReset.addEventListener('click', doReset);
    btnStep.addEventListener('click', doStep);
    btnAuto.addEventListener('click', startAutoRun);
    btnStop.addEventListener('click', stopAutoRun);

    // Initial connection check
    checkConnection();
    setInterval(checkConnection, 5000);

    // API URL change
    apiUrlInput.addEventListener('change', () => {
      checkConnection();
    });

    toast('🚁 Drone Delivery Simulator v2.0 ready – use sliders or arrow keys to control', 'info', 5000);
  }

  // Boot
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
