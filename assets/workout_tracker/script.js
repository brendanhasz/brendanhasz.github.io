function showFatal(text) {
  let banner = document.getElementById('fatal-error');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'fatal-error';
    banner.className = 'fatal-error';
    banner.setAttribute('role', 'alert');
    document.body.prepend(banner);
  }
  banner.textContent = `Something went wrong: ${text}`;
}

window.addEventListener('error', (event) => showFatal(event.message));
window.addEventListener('unhandledrejection', (event) =>
  showFatal(event.reason?.message || String(event.reason))
);

const STORAGE_KEYS = {
  workouts: 'workout-tracker:workouts',
  history: 'workout-tracker:history',
};

const EXERCISE_FIELDS = {
  rest: ['duration'],
  static: ['name', 'variation', 'duration'],
  reps: ['name', 'variation', 'duration', 'reps', 'weight'],
};

const FIELD_DEFS = {
  name: { label: 'Name', type: 'text' },
  variation: { label: 'Variation', type: 'text' },
  duration: { label: 'Duration (seconds)', type: 'number', min: '0', step: '1' },
  reps: { label: 'Reps', type: 'number', min: '1', step: '1' },
  weight: { label: 'Weight', type: 'number', min: '0', step: 'any' },
  repetitions: { label: 'Repetitions', type: 'number', min: '1', step: '1' },
};

const SUPERSET_FIELDS = ['name', 'repetitions'];

const TYPE_LABELS = { rest: 'Rest', static: 'Static', reps: 'Reps', superset: 'Superset' };

const $ = (id) => document.getElementById(id);
const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

function saveList(name, data) {
  localStorage.setItem(STORAGE_KEYS[name], JSON.stringify(data));
}

function loadList(name) {
  let raw;
  try {
    raw = localStorage.getItem(STORAGE_KEYS[name]);
  } catch {
    showFatal('Browser storage is unavailable, so nothing can be saved.');
    return [];
  }
  if (raw === null) return [];
  try {
    const data = JSON.parse(raw);
    if (Array.isArray(data)) return data;
  } catch {
    // Unreadable: fall through and set it aside rather than overwrite it.
  }
  try {
    localStorage.setItem(`${STORAGE_KEYS[name]}:unreadable`, raw);
  } catch {
    // Nothing more can be done if storage is full.
  }
  return [];
}

let workouts = loadList('workouts');
let history = loadList('history');
let editingIndex = null;

function updateAddButton() {
  $('add-workout-button').hidden = !$('workouts-tab').classList.contains('is-active');
}

function formatDuration(totalSeconds) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.round(totalSeconds % 60);
  const parts = [];
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  if (seconds || !parts.length) parts.push(`${seconds}s`);
  return parts.join(' ');
}

function exerciseDuration(exercise) {
  if (exercise.type === 'superset') {
    const roundTime = (exercise.exercises || []).reduce((sum, inner) => sum + exerciseDuration(inner), 0);
    return (Number(exercise.repetitions) || 0) * roundTime;
  }
  return Number(exercise.duration) || 0;
}

function totalDuration(workout) {
  return (workout.exercises || []).reduce((sum, exercise) => sum + exerciseDuration(exercise), 0);
}

// One entry per exercise actually performed; supersets repeat, still pointing at the saved objects.
function flattenExercises(exercises) {
  const flat = [];
  exercises.forEach((exercise) => {
    if (exercise.type !== 'superset') {
      flat.push({ exercise, context: '' });
      return;
    }
    for (let round = 1; round <= exercise.repetitions; round += 1) {
      (exercise.exercises || []).forEach((inner) => {
        flat.push({ exercise: inner, context: `${exercise.name} (round ${round} of ${exercise.repetitions})` });
      });
    }
  });
  return flat;
}

function activateTab(tab, moveFocus = false) {
  tabs.forEach((item) => {
    const isActive = item === tab;
    item.classList.toggle('is-active', isActive);
    item.setAttribute('aria-selected', String(isActive));
    item.tabIndex = isActive ? 0 : -1;
  });

  panels.forEach((panel) => {
    panel.hidden = panel.id !== tab.getAttribute('aria-controls');
  });

  updateAddButton();

  if (moveFocus) {
    tab.focus();
  }
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', (event) => {
    let nextIndex;

    if (event.key === 'ArrowRight') {
      nextIndex = (index + 1) % tabs.length;
    } else if (event.key === 'ArrowLeft') {
      nextIndex = (index - 1 + tabs.length) % tabs.length;
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else if (event.key === 'End') {
      nextIndex = tabs.length - 1;
    } else {
      return;
    }

    event.preventDefault();
    activateTab(tabs[nextIndex], true);
  });
});

function showMessage(text) {
  const message = $('workouts-message');
  message.textContent = text;
  message.hidden = !text;
}

function renderWorkouts() {
  $('workout-list').replaceChildren(
    ...workouts.map((workout, index) => {
      const item = document.createElement('li');
      const row = document.createElement('div');
      row.className = 'workout-row';

      const info = document.createElement('div');
      info.className = 'workout-info';

      const name = document.createElement('span');
      name.className = 'workout-name';
      name.textContent = workout.name;
      const time = document.createElement('span');
      time.className = 'workout-time';
      time.textContent = formatDuration(totalDuration(workout));

      info.append(name);
      const actions = document.createElement('div');
      actions.className = 'workout-actions';

      const edit = document.createElement('button');
      edit.type = 'button';
      edit.className = 'workout-action edit-workout';
      edit.setAttribute('aria-label', `Edit ${workout.name}`);
      edit.title = 'Edit workout';
      edit.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 16.5V20h3.5L19 8.5 15.5 5 4 16.5zM13.5 7l3.5 3.5"/></svg>';
      edit.addEventListener('click', () => openCreateScreen(index));

      const start = document.createElement('button');
      start.type = 'button';
      start.className = 'workout-action start-workout';
      start.setAttribute('aria-label', `Start ${workout.name}`);
      start.title = 'Start workout';
      start.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.8c0-.8.9-1.3 1.6-.9l11.1 7.1a1.2 1.2 0 0 1 0 2L8.6 20.1c-.7.4-1.6-.1-1.6-.9V4.8z"/></svg>';
      start.addEventListener('click', () => startWorkout(workout));

      actions.append(edit, start);
      row.append(info, time, actions);
      item.append(row);
      return item;
    })
  );
  showMessage(workouts.length ? '' : 'No workouts yet. Tap + to create one.');
}

function formatCompletedAt(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? ''
    : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}

function renderHistory() {
  $('history-list').replaceChildren(
    ...history
      .slice()
      .reverse()
      .map((entry) => {
        const item = document.createElement('li');
        const row = document.createElement('div');
        row.className = 'history-row';

        const info = document.createElement('div');
        info.className = 'history-info';
        const name = document.createElement('span');
        name.className = 'workout-name';
        name.textContent = entry.name;
        const date = document.createElement('span');
        date.className = 'history-date';
        date.textContent = formatCompletedAt(entry.completedAt);
        info.append(name, date);

        const time = document.createElement('span');
        time.className = 'workout-time';
        time.textContent = formatDuration(totalDuration(entry));

        row.append(info, time);
        item.append(row);
        return item;
      })
  );
  const message = $('history-message');
  message.textContent = history.length ? '' : 'No completed workouts yet.';
  message.hidden = history.length > 0;
}

function downloadJson(json, fileName) {
  const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const getList = (name) => (name === 'workouts' ? workouts : history);

function applyList(name, data) {
  if (name === 'workouts') {
    workouts = data;
    renderWorkouts();
  } else {
    history = data;
    renderHistory();
  }
}

function setStatus(name, text, isError = false) {
  const status = $(`${name}-status`);
  status.textContent = text;
  status.hidden = !text;
  status.classList.toggle('is-error', isError);
}

function isValidExercise(exercise, allowSuperset = true) {
  if (exercise && exercise.type === 'superset') {
    return Boolean(
      allowSuperset &&
        typeof exercise.name === 'string' &&
        exercise.name &&
        Number.isInteger(exercise.repetitions) &&
        exercise.repetitions >= 1 &&
        Array.isArray(exercise.exercises) &&
        exercise.exercises.length &&
        exercise.exercises.every((inner) => isValidExercise(inner, false))
    );
  }
  return Boolean(
    exercise &&
      Object.hasOwn(EXERCISE_FIELDS, exercise.type) &&
      Number.isFinite(exercise.duration) &&
      exercise.duration >= 0 &&
      (exercise.type === 'rest' || (typeof exercise.name === 'string' && exercise.name)) &&
      (exercise.type !== 'reps' || (Number.isFinite(exercise.reps) && Number.isFinite(exercise.weight)))
  );
}

function isValidEntry(name, item) {
  return Boolean(
    item &&
      typeof item.name === 'string' &&
      item.name &&
      Array.isArray(item.exercises) &&
      item.exercises.every((exercise) => isValidExercise(exercise)) &&
      (name === 'workouts' || typeof item.completedAt === 'string')
  );
}

function exportList(name) {
  downloadJson(`${JSON.stringify(getList(name), null, 2)}\n`, `${name}.json`);
  setStatus(name, `Exported ${plural(getList(name).length, 'item')} to ${name}.json.`);
}

function importList(name) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json,application/json';
  input.addEventListener('change', async () => {
    const file = input.files[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      if (!Array.isArray(data) || !data.every((item) => isValidEntry(name, item))) {
        throw new Error(`That file is not a valid ${name} export.`);
      }
      const current = getList(name);
      const replaceMessage = `Replace your ${plural(current.length, 'saved item')} with the ${data.length} in this file?`;
      if (current.length && !window.confirm(replaceMessage)) return;
      saveList(name, data);
      applyList(name, data);
      setStatus(name, `Imported ${plural(data.length, 'item')}.`);
    } catch (error) {
      setStatus(name, error instanceof SyntaxError ? 'That file is not valid JSON.' : error.message, true);
    }
  });
  input.click();
}

let pendingClear = null;

function closeConfirm() {
  pendingClear = null;
  $('confirm-view').hidden = true;
}

function askClear(name) {
  if (!getList(name).length) {
    setStatus(name, 'Nothing to clear.');
    return;
  }
  pendingClear = name;
  $('confirm-text').textContent = `This will permanently delete all your ${name}. It can't be undone.`;
  $('confirm-view').hidden = false;
  $('confirm-no').focus();
}

function confirmClear() {
  const name = pendingClear;
  closeConfirm();
  if (!name) return;
  try {
    saveList(name, []);
    applyList(name, []);
    setStatus(name, 'Cleared.');
  } catch {
    setStatus(name, 'Could not clear. Browser storage may be blocked.', true);
  }
}

let session = null;

function setWorkoutError(text) {
  const error = $('workout-error');
  error.textContent = text;
  error.hidden = !text;
}

function persistEdits() {
  try {
    saveList('workouts', workouts);
    setWorkoutError('');
  } catch {
    setWorkoutError('Could not save changes. Browser storage may be full or blocked.');
  }
}

function recordCompletion() {
  if (!session.exercises.length) return;

  const entry = {
    name: session.name,
    exercises: JSON.parse(JSON.stringify(session.source)),
    completedAt: new Date().toISOString(),
  };
  const updated = [...history, entry];
  try {
    saveList('history', updated);
    history = updated;
    renderHistory();
  } catch {
    setWorkoutError('Could not record this workout. Browser storage may be full or blocked.');
  }
}

const SLOTS = [
  ['slot-previous', -1, 'No previous exercise'],
  ['slot-current', 0, 'Workout complete'],
  ['slot-next', 1, 'No next exercise'],
];

function formatClock(totalSeconds) {
  const whole = Math.max(0, Math.ceil(totalSeconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

function stepButton(text, label, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'step-button';
  button.textContent = text;
  button.setAttribute('aria-label', label);
  button.addEventListener('click', onClick);
  return button;
}

function stepper(label, exercise, key, step) {
  const group = document.createElement('div');
  group.className = 'stepper';
  const caption = document.createElement('span');
  caption.className = 'stepper-label';
  caption.textContent = label;
  const value = document.createElement('span');
  value.className = 'stepper-value';
  value.textContent = exercise[key];

  const change = (delta) => {
    exercise[key] = Math.max(0, Number(((Number(exercise[key]) || 0) + delta).toFixed(2)));
    value.textContent = exercise[key];
    persistEdits();
  };

  const controls = document.createElement('div');
  controls.className = 'stepper-controls';
  controls.append(
    stepButton('\u2212', `Decrease ${label.toLowerCase()}`, () => change(-step)),
    value,
    stepButton('+', `Increase ${label.toLowerCase()}`, () => change(step))
  );
  group.append(caption, controls);
  return group;
}

function buildExercise(exercise, context) {
  const row = document.createElement('div');
  row.className = 'exercise-row';

  const main = document.createElement('div');
  main.className = 'exercise-main';
  if (context) {
    const label = document.createElement('span');
    label.className = 'exercise-context';
    label.textContent = context;
    main.append(label);
  }
  const name = document.createElement('span');
  name.className = 'exercise-name';
  name.textContent = exercise.type === 'rest' ? 'Rest' : exercise.name;
  main.append(name);
  if (exercise.variation) {
    const variation = document.createElement('span');
    variation.className = 'exercise-variation';
    variation.textContent = exercise.variation;
    main.append(variation);
  }

  const time = document.createElement('span');
  time.className = 'exercise-time';
  time.textContent = formatClock(exercise.duration);
  row.append(main, time);

  if (exercise.type === 'reps') {
    const stats = document.createElement('div');
    stats.className = 'exercise-stats';
    stats.append(stepper('Reps', exercise, 'reps', 1), stepper('Weight', exercise, 'weight', 1));
    row.append(stats);
  }
  return { row, time };
}

function renderSlots() {
  session.timeEl = null;
  if (session.index === -1) {
    const ready = document.createElement('div');
    ready.className = 'exercise-row start-countdown';
    const title = document.createElement('span');
    title.className = 'exercise-name';
    title.textContent = 'Starting in';
    const time = document.createElement('span');
    time.className = 'exercise-time';
    time.textContent = formatClock(-session.elapsed);
    ready.append(title, time);
    $('slot-current').replaceChildren(ready);
    session.timeEl = time;
  }
  SLOTS.forEach(([id, offset, emptyText]) => {
    if (session.index === -1 && id === 'slot-current') return;
    const position = session.index + offset;
    const exercise = session.exercises[position];
    if (!exercise) {
      const empty = document.createElement('p');
      empty.className = 'slot-empty';
      empty.textContent = emptyText;
      $(id).replaceChildren(empty);
      return;
    }
    const { row, time } = buildExercise(exercise, session.contexts[position]);
    if (offset === 0) session.timeEl = time;
    $(id).replaceChildren(row);
  });
}

function setPaused(paused) {
  $('pause-button').classList.toggle('is-paused', paused);
  $('pause-button').setAttribute('aria-label', paused ? 'Resume workout' : 'Pause workout');
}

function stopWorkout() {
  if (!session) return;
  clearInterval(session.timerId);
  cancelSpeech();
  session = null;
}

function finishWorkout() {
  clearInterval(session.timerId);
  session.running = false;
  $('pause-button').hidden = true;
  recordCompletion();
}

const SPOKEN_NUMBERS = {
  1: 'One',
  2: 'Two',
  3: 'Three',
  10: 'Ten',
  20: 'Twenty',
  30: 'Thirty',
};

const hasSpeech = 'speechSynthesis' in window;
const activeUtterances = new Set();
let currentSpeech = null;

function speak(text, { interrupt = false } = {}) {
  if (!hasSpeech) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.isCountdown = interrupt;
  // Holding a reference stops Chrome from garbage-collecting the utterance mid-speech.
  activeUtterances.add(utterance);
  utterance.onstart = () => {
    currentSpeech = utterance;
  };
  utterance.onend = utterance.onerror = () => {
    activeUtterances.delete(utterance);
    if (currentSpeech === utterance) currentSpeech = null;
  };
  if (window.speechSynthesis.paused) window.speechSynthesis.resume();
  // Speech never overlaps, so a new countdown number cuts off the previous one instead of queueing behind it.
  if (interrupt && currentSpeech?.isCountdown) window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function cancelSpeech() {
  if (hasSpeech) window.speechSynthesis.cancel();
}

const plural = (count, word) => `${count} ${word}${count === 1 ? '' : 's'}`;

// Speech starts late, so the 3-2-1 countdown is spoken this much before each number shows.
const COUNTDOWN_LEAD_SECONDS = 0.25;

function formatSpokenDuration(totalSeconds) {
  const whole = Math.round(totalSeconds);
  const minutes = Math.floor(whole / 60);
  const seconds = whole % 60;
  const parts = [];
  if (minutes) parts.push(plural(minutes, 'minute'));
  if (seconds || !minutes) parts.push(plural(seconds, 'second'));
  return parts.join(' and ');
}

// Cues only fire when the whole-second remainder hits the mark, so a stale cue is never spoken late.
function announceTimedExercise(index) {
  const exercise = session.exercises[index];
  if (exercise.type !== 'rest' && exercise.type !== 'static') return;

  if (session.announcedIndex !== index) {
    session.announcedIndex = index;
    const duration = Number(exercise.duration) || 0;
    // The opening announcement already covers cues that would fire at or before the start.
    session.spoken = new Set([Math.ceil(duration)]);
    [1, 2, 3].forEach((n) => {
      if (n + COUNTDOWN_LEAD_SECONDS >= duration) session.spoken.add(n);
    });
    speak(`${exercise.type === 'rest' ? 'Rest' : exercise.name} for ${formatSpokenDuration(duration)}`);
    return;
  }

  const remaining = session.ends[index] - session.elapsed;
  const left = Math.ceil(remaining);
  const countdown = Math.ceil(remaining - COUNTDOWN_LEAD_SECONDS);
  if (countdown >= 1 && countdown <= 3) {
    if (!session.spoken.has(countdown)) {
      session.spoken.add(countdown);
      speak(SPOKEN_NUMBERS[countdown], { interrupt: true });
    }
    return;
  }

  if (session.spoken.has(left)) return;
  if (left === 10 || left === 20 || left === 30) {
    session.spoken.add(left);
    speak(`${SPOKEN_NUMBERS[left]} seconds left`);
  } else if (left > 30 && left % 60 === 0) {
    session.spoken.add(left);
    speak(`${plural(left / 60, 'minute')} left`);
  }
}

function tick() {
  if (!session.running) return;
  const now = performance.now();
  session.elapsed += (now - session.last) / 1000;
  session.last = now;

  const found = session.ends.findIndex((end) => end > session.elapsed);
  const index = session.elapsed < 0 ? -1 : found === -1 ? session.exercises.length : found;
  if (index !== session.index) {
    session.index = index;
    renderSlots();
  }

  if (index === -1) {
    const remaining = -session.elapsed;
    session.timeEl.textContent = formatClock(remaining);
    const countdown = Math.ceil(remaining - COUNTDOWN_LEAD_SECONDS);
    if (countdown >= 1 && countdown <= 3 && !session.startSpoken.has(countdown)) {
      session.startSpoken.add(countdown);
      speak(SPOKEN_NUMBERS[countdown], { interrupt: true });
    }
  } else if (index === session.exercises.length) {
    finishWorkout();
  } else {
    session.timeEl.textContent = formatClock(session.ends[index] - session.elapsed);
    announceTimedExercise(index);
  }
}

function startWorkout(workout) {
  stopWorkout();
  const flat = flattenExercises(workout.exercises || []);
  const exercises = flat.map((item) => item.exercise);
  let total = 0;
  const ends = exercises.map((exercise) => (total += Number(exercise.duration) || 0));

  session = {
    name: workout.name,
    source: workout.exercises || [],
    exercises,
    contexts: flat.map((item) => item.context),
    ends,
    elapsed: -10,
    index: -2,
    running: true,
    last: performance.now(),
    timeEl: null,
    announcedIndex: -1,
    spoken: new Set(),
    startSpoken: new Set(),
    timerId: setInterval(tick, 100),
  };
  $('workout-title').textContent = workout.name;
  setWorkoutError('');
  $('pause-button').hidden = false;
  setPaused(false);
  showView('workout');
  tick();
  speak(`${workout.name} starting in ten seconds`);
}

function togglePause() {
  if (!session) return;
  if (session.running) {
    tick();
    session.running = false;
    cancelSpeech();
  } else {
    session.last = performance.now();
    session.running = true;
  }
  setPaused(!session.running);
}

function showView(name) {
  $('main-view').hidden = name !== 'main';
  $('create-view').hidden = name !== 'create';
  $('workout-view').hidden = name !== 'workout';
}

function setFormError(text) {
  const error = $('form-error');
  error.textContent = text;
  error.hidden = !text;
}

function buildField(key) {
  const def = FIELD_DEFS[key];
  const label = document.createElement('label');
  label.className = 'field';
  const caption = document.createElement('span');
  caption.textContent = def.label;
  const input = document.createElement('input');
  input.type = def.type;
  input.dataset.field = key;
  input.autocomplete = 'off';
  if (def.type === 'number') {
    input.min = def.min;
    input.step = def.step;
    input.inputMode = 'decimal';
  } else {
    input.maxLength = 80;
  }
  label.append(caption, input);
  return label;
}

// Every exercise list alternates slot, card, slot, card, ... slot, so a plus sits at every position.
function buildInsertSlot(list) {
  const slot = document.createElement('li');
  slot.className = 'insert-slot';

  const plus = document.createElement('button');
  plus.type = 'button';
  plus.className = 'add-exercise-button';
  plus.setAttribute('aria-label', 'Add exercise here');
  plus.setAttribute('aria-expanded', 'false');
  plus.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>';

  const picker = document.createElement('div');
  picker.className = 'type-picker';
  picker.hidden = true;
  const types = list.classList.contains('superset-list') ? ['rest', 'static', 'reps'] : Object.keys(TYPE_LABELS);
  types.forEach((type) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.type = type;
    button.textContent = TYPE_LABELS[type];
    picker.append(button);
  });

  const setOpen = (open) => {
    picker.hidden = !open;
    plus.setAttribute('aria-expanded', String(open));
  };
  plus.addEventListener('click', () => setOpen(picker.hidden));
  picker.addEventListener('click', (event) => {
    const { type } = event.target.dataset;
    if (!type) return;
    setOpen(false);
    addExerciseCard(type, list, slot);
  });

  slot.append(plus, picker);
  return slot;
}

function addExerciseCard(type, list, slot, values = null) {
  const card = document.createElement('li');
  card.className = 'exercise-card';
  card.dataset.type = type;

  const header = document.createElement('div');
  header.className = 'exercise-card-header';
  const title = document.createElement('strong');
  title.textContent = TYPE_LABELS[type];
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'remove-exercise';
  remove.textContent = 'Remove';
  remove.addEventListener('click', () => {
    card.nextElementSibling.remove();
    card.remove();
  });
  header.append(title, remove);
  card.append(header);

  const keys = type === 'superset' ? SUPERSET_FIELDS : EXERCISE_FIELDS[type];
  keys.forEach((key) => {
    const field = buildField(key);
    if (values && values[key] !== undefined) field.querySelector('input').value = values[key];
    card.append(field);
  });

  if (type === 'superset') {
    card.classList.add('superset-card');
    const inner = document.createElement('ol');
    inner.className = 'exercise-list superset-list';
    inner.append(buildInsertSlot(inner));
    card.append(inner);
    if (values) populateExerciseList(inner, values.exercises || []);
    else card.querySelector('[data-field="repetitions"]').value = '2';
  }

  slot.after(card, buildInsertSlot(list));
  if (!values) card.querySelector('input').focus();
}

function populateExerciseList(list, exercises) {
  exercises.forEach((exercise) => {
    addExerciseCard(exercise.type, list, list.lastElementChild, exercise);
  });
}

function readInput(input, exercise) {
  const key = input.dataset.field;
  exercise[key] = input.type === 'number' ? Number(input.value) || 0 : input.value.trim();
}

const cardsIn = (list) => Array.from(list.querySelectorAll(':scope > .exercise-card'));

function readExercise(card) {
  const exercise = { type: card.dataset.type };
  if (exercise.type === 'superset') {
    card.querySelectorAll(':scope > .field input').forEach((input) => readInput(input, exercise));
    exercise.exercises = cardsIn(card.querySelector(':scope > .superset-list')).map(readExercise);
    return exercise;
  }
  card.querySelectorAll('input').forEach((input) => readInput(input, exercise));
  return exercise;
}

function findExerciseError(exercises) {
  for (const exercise of exercises) {
    if (exercise.type === 'superset') {
      if (!exercise.name) return 'Every superset needs a name.';
      if (!Number.isInteger(exercise.repetitions) || exercise.repetitions < 1) {
        return 'Superset repetitions must be a whole number of at least 1.';
      }
      if (!exercise.exercises.length) return 'Every superset needs at least one exercise.';
      const inner = findExerciseError(exercise.exercises);
      if (inner) return inner;
    } else if (exercise.type !== 'rest' && !exercise.name) {
      return 'Every Static and Reps exercise needs a name.';
    }
  }
  return '';
}

function openCreateScreen(index = null) {
  editingIndex = index;
  const workout = index === null ? null : workouts[index];
  $('workout-form').reset();
  const list = $('exercise-list');
  list.replaceChildren(buildInsertSlot(list));
  $('create-title').textContent = workout ? 'Edit workout' : 'Create new workout';
  $('workout-name').value = workout ? workout.name : '';
  if (workout) populateExerciseList(list, workout.exercises);
  setFormError('');
  showView('create');
  if (!workout) $('workout-name').focus();
}

function saveWorkout() {
  const name = $('workout-name').value.trim();
  if (!name) {
    setFormError('Give the workout a name.');
    return;
  }

  const exercises = cardsIn($('exercise-list')).map(readExercise);
  const exerciseError = findExerciseError(exercises);
  if (exerciseError) {
    setFormError(exerciseError);
    return;
  }

  const updated = [...workouts];
  const workout = { name, exercises };
  if (editingIndex === null) updated.push(workout);
  else updated[editingIndex] = workout;
  try {
    saveList('workouts', updated);
  } catch {
    setFormError('Could not save the workout. Browser storage may be full or blocked.');
    return;
  }

  workouts = updated;
  editingIndex = null;
  renderWorkouts();
  showView('main');
}

$('add-workout-button').addEventListener('click', openCreateScreen);
$('back-button').addEventListener('click', () => showView('main'));
$('workout-back-button').addEventListener('click', () => {
  stopWorkout();
  showView('main');
});
$('pause-button').addEventListener('click', togglePause);
$('save-button').addEventListener('click', saveWorkout);
$('workout-form').addEventListener('submit', (event) => event.preventDefault());

document.querySelectorAll('[data-action="export"]').forEach((button) => {
  button.addEventListener('click', () => exportList(button.dataset.store));
});
document.querySelectorAll('[data-action="import"]').forEach((button) => {
  button.addEventListener('click', () => importList(button.dataset.store));
});
document.querySelectorAll('[data-action="clear"]').forEach((button) => {
  button.addEventListener('click', () => askClear(button.dataset.store));
});
$('confirm-no').addEventListener('click', closeConfirm);
$('confirm-yes').addEventListener('click', confirmClear);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !$('confirm-view').hidden) closeConfirm();
});

renderWorkouts();
renderHistory();
updateAddButton();