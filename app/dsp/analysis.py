import numpy as np
import librosa

GHOST_THRESHOLD = 0.15
REPEAT_SIM_THRESHOLD = 0.85
CHORD_CONF_FLOOR = 0.5
CHORD_MIN_DURATION = 1.5

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
BANDS = {'low': (20, 250), 'mid': (250, 4000), 'high': (4000, None)}

DRUM_PHRASES = ['약한 하이햇 패턴', '기본 드럼 루프', '풀 드럼 킷 진행']
BASS_PHRASES = ['서브 저음 등장', '기본 베이스 라인', '움직임이 큰 베이스 라인']
VOCAL_PHRASES = ['보컬 여백 구간', '보컬 진행', '보컬 에너지 강조 구간']
FX_PHRASES = ['공간감 FX', '고역 텍스처', '고역 FX 확장']


def _build_chord_templates():
    major = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    minor = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    labels, vectors = [], []
    for i, name in enumerate(NOTE_NAMES):
        labels.append(name)
        vectors.append(np.roll(major, i))
        labels.append(name + 'm')
        vectors.append(np.roll(minor, i))
    matrix = np.stack(vectors)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return labels, matrix


CHORD_LABELS, CHORD_MATRIX = _build_chord_templates()


def _zscore(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def _normalize_0_1(values):
    values = np.asarray(values, dtype=float)
    peak = values.max() if values.size else 0.0
    if peak <= 1e-9:
        return np.zeros_like(values)
    return np.clip(values / peak, 0, 1)


def _estimate_chords(chroma):
    norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
    sims = CHORD_MATRIX @ norm
    best_idx = np.argmax(sims, axis=0)
    best_sim = sims[best_idx, np.arange(sims.shape[1])]
    return [CHORD_LABELS[i] if s >= CHORD_CONF_FLOOR else 'N' for i, s in zip(best_idx, best_sim)]


def _smooth_chord_sequence(labels, boundaries, min_duration):
    n = len(labels)
    if n == 0:
        return []
    runs = []
    cur_label, cur_start = labels[0], float(boundaries[0])
    for i in range(1, n):
        if labels[i] != cur_label:
            runs.append([cur_label, cur_start, float(boundaries[i])])
            cur_label, cur_start = labels[i], float(boundaries[i])
    runs.append([cur_label, cur_start, float(boundaries[n])])

    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if run[2] - run[1] < min_duration:
                if i > 0:
                    runs[i - 1][2] = run[2]
                else:
                    runs[1][1] = run[1]
                del runs[i]
                changed = True
                break

    merged = [runs[0]]
    for run in runs[1:]:
        if run[0] == merged[-1][0]:
            merged[-1][2] = run[2]
        else:
            merged.append(run)
    return merged


def _chords_in_range(chord_runs, start, end):
    names = []
    for label, s, e in chord_runs:
        if e <= start or s >= end or label == 'N':
            continue
        if not names or names[-1] != label:
            names.append(label)
    return names


def _segment_song(y, y_harm, sr, duration):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0

    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    n_frames = chroma.shape[1]

    use_beats = len(beats) >= max(8, duration / 3)
    if use_beats:
        markers = librosa.util.fix_frames(beats, x_min=0, x_max=n_frames)
    else:
        hop_frames = max(1, int(round(2 * sr / 512)))
        markers = librosa.util.fix_frames(np.arange(0, n_frames, hop_frames), x_min=0, x_max=n_frames)

    marker_times = librosa.frames_to_time(markers, sr=sr)
    if len(marker_times):
        marker_times = np.append(marker_times[:-1], duration)

    chroma_sync = librosa.util.sync(chroma, markers, aggregate=np.median)
    n_units = chroma_sync.shape[1]

    if n_units < 4:
        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
        vec = chroma_sync.mean(axis=1) if n_units else np.zeros(12)
        segments = [{'start': 0.0, 'end': duration, 'chroma': vec, 'rms': rms}]
        return tempo, segments, chroma_sync, marker_times

    mfcc_sync = librosa.util.sync(mfcc, markers, aggregate=np.mean)
    features = np.vstack([_zscore(chroma_sync), _zscore(mfcc_sync)])

    k = int(np.clip(round(duration / 20), 4, 12))
    k = min(k, max(2, n_units - 1))
    try:
        boundary_units = [int(b) for b in librosa.segment.agglomerative(features, k)]
    except Exception:
        boundary_units = [int(b) for b in np.linspace(0, n_units, num=k, endpoint=False)]

    boundary_units = sorted(set([0, *boundary_units, n_units]))

    segments = []
    for i in range(len(boundary_units) - 1):
        u0, u1 = boundary_units[i], boundary_units[i + 1]
        if u1 <= u0:
            continue
        t0, t1 = float(marker_times[u0]), float(marker_times[u1])
        s0, s1 = int(t0 * sr), int(t1 * sr)
        seg_audio = y[s0:s1]
        rms = float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) else 0.0
        vec = chroma_sync[:, u0:u1].mean(axis=1)
        segments.append({'start': t0, 'end': t1, 'chroma': vec, 'rms': rms})

    if segments:
        segments[-1]['end'] = duration
    else:
        segments = [{'start': 0.0, 'end': duration, 'chroma': chroma_sync.mean(axis=1), 'rms': 0.0}]

    return tempo, segments, chroma_sync, marker_times


def _group_repeats(segments):
    n = len(segments)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    vecs = np.stack([s['chroma'] for s in segments])
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= REPEAT_SIM_THRESHOLD:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return groups


def _label_sections(segments):
    n = len(segments)
    groups = _group_repeats(segments)
    segment_root = [None] * n
    for root, idxs in groups.items():
        for i in idxs:
            segment_root[i] = root

    repeated_roots = [r for r, idxs in groups.items() if len(idxs) >= 2]
    if not repeated_roots:
        return [f'Section {i + 1}' for i in range(n)]

    mean_rms = {r: float(np.mean([segments[i]['rms'] for i in groups[r]])) for r in repeated_roots}
    chorus_root = max(repeated_roots, key=lambda r: mean_rms[r])
    chorus_members = set(groups[chorus_root])

    other_repeated = sorted(
        [r for r in repeated_roots if r != chorus_root],
        key=lambda r: min(groups[r]),
    )
    role_pool = ['Verse', 'Pre-chorus']
    role_for_root = {r: (role_pool[i] if i < len(role_pool) else f'Section-{i + 1}')
                      for i, r in enumerate(other_repeated)}

    avg_seg_len = (segments[-1]['end'] / n) if n else 0
    counters = {}
    labels = []
    for i in range(n):
        root = segment_root[i]
        seg_len = segments[i]['end'] - segments[i]['start']

        if root == chorus_root:
            base = 'Chorus'
        elif root in role_for_root:
            base = role_for_root[root]
        elif i == 0 and seg_len <= avg_seg_len * 1.3:
            labels.append('Intro')
            continue
        elif i == n - 1 and seg_len <= avg_seg_len * 1.3:
            labels.append('Outro')
            continue
        elif (i + 1) < n and (i + 1) in chorus_members:
            base = 'Pre-chorus'
        else:
            base = 'Bridge'

        counters[base] = counters.get(base, 0) + 1
        labels.append(f'{base} {counters[base]}')

    return labels


def _finalize_sections(segments, labels, duration):
    starts = [int(round(s['start'])) for s in segments]
    starts[0] = 0
    sections = []
    for i in range(len(segments)):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(segments) else int(round(duration))
        if end <= start:
            end = start + 1
        sections.append({'label': labels[i], 'start': start, 'end': end})
    sections[-1]['end'] = int(round(duration))
    return sections


def _compute_chords(beat_chroma, beat_times, duration):
    n_units = beat_chroma.shape[1]
    if n_units == 0:
        return []
    chord_seq = _estimate_chords(beat_chroma)
    if len(beat_times) >= n_units + 1:
        boundaries = beat_times[:n_units + 1]
    else:
        boundaries = np.append(beat_times[:n_units], duration)
    runs = _smooth_chord_sequence(chord_seq, boundaries, CHORD_MIN_DURATION)
    if runs:
        runs[-1][2] = duration
    return runs


def _harmony_top_bottom(sections, chord_runs):
    top, bottom = [], []
    for sec in sections:
        names = _chords_in_range(chord_runs, sec['start'], sec['end'])
        if not names:
            continue
        entry = {'s': sec['start'], 'e': sec['end'], 'label': ' - '.join(names)}
        if 'Chorus' in sec['label']:
            bottom.append(entry)
        else:
            top.append(entry)
    return top, bottom


def _band_energy_per_section(y, sr, sections):
    n_fft, hop = 2048, 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)

    nyquist = sr / 2
    band_masks = {}
    for name, (lo, hi) in BANDS.items():
        hi = hi if hi is not None else nyquist
        band_masks[name] = (freqs >= lo) & (freqs < hi)

    raw = {name: [] for name in BANDS}
    for sec in sections:
        frame_mask = (frame_times >= sec['start']) & (frame_times < sec['end'])
        if not np.any(frame_mask):
            nearest = int(np.abs(frame_times - sec['start']).argmin())
            frame_mask = np.arange(S.shape[1]) == nearest
        for name, bmask in band_masks.items():
            if np.any(bmask) and np.any(frame_mask):
                val = float(np.mean(S[np.ix_(bmask, frame_mask)]))
            else:
                val = 0.0
            raw[name].append(val)

    normalized = {name: _normalize_0_1(raw[name]) for name in BANDS}
    freq_map = {}
    for i, sec in enumerate(sections):
        freq_map[sec['label']] = [
            round(float(normalized['low'][i]), 2),
            round(float(normalized['mid'][i]), 2),
            round(float(normalized['high'][i]), 2),
        ]
    return freq_map


def _band_rms_series(signal, sr, lo, hi, n_fft=2048, hop=512):
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hi = hi if hi is not None else sr / 2
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(S.shape[1])
    return np.sqrt(np.mean(S[mask] ** 2, axis=0))


def _track_segments(y, y_harm, y_perc, sr, sections, chord_runs):
    hop, n_fft = 512, 2048

    drum_series = librosa.feature.rms(y=y_perc, frame_length=n_fft, hop_length=hop)[0]
    bass_series = _band_rms_series(y, sr, *BANDS['low'])
    chord_series = _band_rms_series(y_harm, sr, 250, 4000)
    vocal_series = _band_rms_series(y_harm, sr, 200, 3400)
    fx_series = _band_rms_series(y, sr, BANDS['high'][0], None)

    frame_times = librosa.frames_to_time(np.arange(len(drum_series)), sr=sr, hop_length=hop)

    def section_level(series, sec):
        mask = (frame_times >= sec['start']) & (frame_times < sec['end'])
        if not np.any(mask):
            return 0.0
        return float(np.mean(series[mask]))

    raw = {
        'Drums': [section_level(drum_series, s) for s in sections],
        'Bass': [section_level(bass_series, s) for s in sections],
        'Chords': [section_level(chord_series, s) for s in sections],
        'Vocal': [section_level(vocal_series, s) for s in sections],
        'FX': [section_level(fx_series, s) for s in sections],
    }
    normalized = {name: _normalize_0_1(vals) for name, vals in raw.items()}

    tracks = []
    for name, phrases in (
        ('Drums', DRUM_PHRASES), ('Bass', BASS_PHRASES),
        ('Chords', None), ('Vocal', VOCAL_PHRASES), ('FX', FX_PHRASES),
    ):
        segs = []
        for i, sec in enumerate(sections):
            level = float(normalized[name][i])
            ghost = level < GHOST_THRESHOLD
            if name == 'Chords':
                chord_names = _chords_in_range(chord_runs, sec['start'], sec['end'])
                label = ('코드 진행: ' + ' - '.join(chord_names)) if chord_names else '코드 진행 불명확'
            else:
                tier = 0 if level < 0.35 else (1 if level < 0.7 else 2)
                tier = min(tier, len(phrases) - 1)
                label = phrases[tier]
            segs.append({'s': sec['start'], 'e': sec['end'], 'label': label, 'ghost': bool(ghost)})
        tracks.append({'name': name, 'segs': segs})
    return tracks


def _fmt_time(seconds):
    seconds = int(seconds)
    return f'{seconds // 60}:{seconds % 60:02d}'


def _build_note_text(tempo, sections, freq_map):
    if not sections:
        return '분석할 수 있는 구간을 찾지 못했어요.'
    energies = [sum(freq_map.get(s['label'], [0, 0, 0])) for s in sections]
    max_i = int(np.argmax(energies))
    min_i = int(np.argmin(energies))
    max_sec, min_sec = sections[max_i], sections[min_i]
    return (
        f"로컬 신호분석 결과 이 곡은 약 {round(tempo)} BPM, {len(sections)}개 구간으로 구성돼요. "
        f"에너지가 가장 높은 구간은 {max_sec['label']}({_fmt_time(max_sec['start'])}~{_fmt_time(max_sec['end'])})이고, "
        f"가장 낮은 구간은 {min_sec['label']}({_fmt_time(min_sec['start'])}~{_fmt_time(min_sec['end'])})이에요. "
        f"(로컬 DSP 분석: 실제 화성/보컬 분리 없이 신호 특징으로 추정한 결과예요.)"
    )


def analyze_audio(y, sr):
    y = librosa.util.normalize(y.astype(np.float32))
    duration = len(y) / sr
    if duration < 5:
        raise ValueError('오디오가 너무 짧아요 (5초 이상 필요).')

    y_harm, y_perc = librosa.effects.hpss(y)

    tempo, segments, beat_chroma, beat_times = _segment_song(y, y_harm, sr, duration)
    labels = _label_sections(segments)
    sections = _finalize_sections(segments, labels, duration)

    chord_runs = _compute_chords(beat_chroma, beat_times, duration)
    freq_map = _band_energy_per_section(y, sr, sections)
    tracks = _track_segments(y, y_harm, y_perc, sr, sections, chord_runs)
    top, bottom = _harmony_top_bottom(sections, chord_runs)
    note_text = _build_note_text(tempo, sections, freq_map)

    return {
        'duration': round(float(duration), 2),
        'tempo': round(float(tempo), 1),
        'sections': [{'label': s['label'], 'start': s['start'], 'end': s['end']} for s in sections],
        'tracks': tracks,
        'harmony': {
            'top': top,
            'bottom': bottom,
            'freq': freq_map,
            'noteText': note_text,
        },
    }
