window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

const DEFAULT_PLOT_DATA_FILES = {
    cloud: 'data/web_cloud_MOR_dij_a0p900_b0p100_20260321_152739.json',
    trajectories: 'data/web_trajectories_MOR_dij_a0p900_b0p100_20260321_152739.json'
};

const PLOTLY_CONFIG = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: {
        format: 'png',
        filename: 'planner_plot',
        scale: 2
    }
};

const COST_COLOR_SCALE = [
    [0.0, '#1E2A78'],
    [0.2, '#2449A4'],
    [0.4, '#1B74B7'],
    [0.6, '#1E9DB7'],
    [0.8, '#33BEA0'],
    [1.0, '#58D27F']
];

const CAM_ELEV = -10;
const CAM_AZIM = 36;
const CAM_ROLL = 90;
const CAM_DISTANCE = 2.8;

const CLOUD_POINT_SIZE = 3;
const CLOUD_POINT_OPACITY = 1.0;
const PATH_Z_OFFSET = 0.0015;
const PATH_LINE_WIDTH = 7;
const PATH_LINE_OUTLINE_WIDTH = 11;

let plannerManifestCache = null;
let plannerUiBound = false;

function degToRad(angle) {
    return angle * Math.PI / 180;
}

function vecNorm(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function vecNormalize(v) {
    const n = vecNorm(v);
    if (n < 1e-9) return [0, 0, 0];
    return [v[0] / n, v[1] / n, v[2] / n];
}

function vecCross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function vecDot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function rotateVectorAroundAxis(v, axisUnit, angleRad) {
    const c = Math.cos(angleRad);
    const s = Math.sin(angleRad);
    const kDotV = vecDot(axisUnit, v);
    const kCrossV = vecCross(axisUnit, v);

    return [
        v[0] * c + kCrossV[0] * s + axisUnit[0] * kDotV * (1 - c),
        v[1] * c + kCrossV[1] * s + axisUnit[1] * kDotV * (1 - c),
        v[2] * c + kCrossV[2] * s + axisUnit[2] * kDotV * (1 - c)
    ];
}

function cameraFromMatplotlib(elevDeg, azimDeg, rollDeg, distance) {
    const el = degToRad(elevDeg);
    const az = degToRad(azimDeg);
    const roll = degToRad(rollDeg);

    const eye = [
        distance * Math.cos(el) * Math.cos(az),
        distance * Math.cos(el) * Math.sin(az),
        distance * Math.sin(el)
    ];

    const forward = vecNormalize([-eye[0], -eye[1], -eye[2]]);
    const worldUp = [0, 0, 1];

    let right = vecNormalize(vecCross(forward, worldUp));
    if (vecNorm(right) < 1e-9) {
        right = vecNormalize(vecCross(forward, [0, 1, 0]));
    }

    const upBase = vecNormalize(vecCross(right, forward));
    const up = vecNormalize(rotateVectorAroundAxis(upBase, forward, roll));

    return {
        eye: { x: eye[0], y: eye[1], z: eye[2] },
        center: { x: 0, y: 0, z: 0 },
        up: { x: up[0], y: up[1], z: up[2] }
    };
}

function setPlannerPlotsStatus(message, isError) {
    const status = document.getElementById('planner-plots-status');
    if (!status) return;

    status.textContent = message;
    if (isError) {
        status.classList.add('is-error');
    } else {
        status.classList.remove('is-error');
    }
}

function normalizeLegacyManifest(dataSources) {
    const cloud = dataSources && dataSources.cloud ? dataSources.cloud : DEFAULT_PLOT_DATA_FILES.cloud;
    const trajectories = dataSources && dataSources.trajectories ? dataSources.trajectories : DEFAULT_PLOT_DATA_FILES.trajectories;

    return {
        version: 1,
        active_study: 'default',
        active_experiment: 'default_run',
        studies: {
            'default': {
                label: 'Default Study',
                fixed_params: {},
                experiments: [
                    {
                        id: 'default_run',
                        label: 'Default Run',
                        cloud: cloud,
                        trajectories: trajectories,
                        params: {}
                    }
                ]
            }
        }
    };
}

function normalizePlotManifest(rawManifest) {
    if (!rawManifest || typeof rawManifest !== 'object') {
        return normalizeLegacyManifest(DEFAULT_PLOT_DATA_FILES);
    }

    if (rawManifest.cloud && rawManifest.trajectories) {
        return normalizeLegacyManifest(rawManifest);
    }

    if (!rawManifest.studies || typeof rawManifest.studies !== 'object') {
        return normalizeLegacyManifest(DEFAULT_PLOT_DATA_FILES);
    }

    const studies = {};
    Object.keys(rawManifest.studies).forEach(function(studyKey) {
        const sourceStudy = rawManifest.studies[studyKey] || {};
        const sourceExperiments = Array.isArray(sourceStudy.experiments) ? sourceStudy.experiments : [];
        const experiments = [];

        sourceExperiments.forEach(function(experiment, index) {
            if (!experiment || !experiment.cloud || !experiment.trajectories) return;

            experiments.push({
                id: String(experiment.id || ('exp_' + index)),
                label: String(experiment.label || experiment.id || ('Experiment ' + (index + 1))),
                cloud: String(experiment.cloud),
                trajectories: String(experiment.trajectories),
                params: (experiment.params && typeof experiment.params === 'object') ? experiment.params : {},
                enabled: experiment.enabled !== false
            });
        });

        studies[studyKey] = {
            label: String(sourceStudy.label || studyKey),
            description: String(sourceStudy.description || ''),
            fixed_params: (sourceStudy.fixed_params && typeof sourceStudy.fixed_params === 'object') ? sourceStudy.fixed_params : {},
            experiments: experiments
        };
    });

    return {
        version: rawManifest.version || 2,
        active_study: rawManifest.active_study || '',
        active_experiment: rawManifest.active_experiment || '',
        studies: studies
    };
}

function studyKeys(manifest) {
    return Object.keys((manifest && manifest.studies) ? manifest.studies : {});
}

function visibleExperiments(study) {
    const all = Array.isArray(study && study.experiments) ? study.experiments : [];
    const enabled = all.filter(function(experiment) {
        return experiment && experiment.enabled !== false;
    });
    return enabled.length > 0 ? enabled : all;
}

function appendSelectOption(selectEl, value, label) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    selectEl.appendChild(option);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatParamValue(value) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value.toFixed(3);
    }
    return String(value);
}

function parameterSortOrder(name) {
    const order = [
        'alpha_g',
        'beta_g',
        'push_ratio',
        'push_alpha',
        'push_beta',
        'path_ratio',
        'path_sigma_scale',
        'sigma_gain',
        'sigma_bias',
        'sigma_eta'
    ];
    const idx = order.indexOf(String(name));
    return idx === -1 ? 999 : idx;
}

function sortedParamEntries(paramsObj) {
    const source = (paramsObj && typeof paramsObj === 'object') ? paramsObj : {};
    return Object.keys(source)
        .sort(function(a, b) {
            const da = parameterSortOrder(a);
            const db = parameterSortOrder(b);
            if (da !== db) return da - db;
            return a.localeCompare(b);
        })
        .map(function(key) {
            return [key, source[key]];
        });
}

function buildParamsRowsHtml(studyFixedParams, experimentParams) {
    const rows = [];
    sortedParamEntries(studyFixedParams).forEach(function(entry) {
        rows.push({
            scope: 'Fixed',
            key: entry[0],
            value: entry[1]
        });
    });
    sortedParamEntries(experimentParams).forEach(function(entry) {
        rows.push({
            scope: 'Experiment',
            key: entry[0],
            value: entry[1]
        });
    });

    if (rows.length === 0) {
        return '<tr><td class="planner-param-value" colspan="3">No parameters declared.</td></tr>';
    }

    return rows.map(function(row) {
        return '<tr>' +
            '<td class="planner-param-name">' + escapeHtml(row.key) + '</td>' +
            '<td class="planner-param-value">' + escapeHtml(formatParamValue(row.value)) + '</td>' +
            '<td class="planner-param-scope">' + escapeHtml(row.scope) + '</td>' +
            '</tr>';
    }).join('');
}

function setPlannerExperimentSummary(study, experiment) {
    const summary = document.getElementById('planner-experiment-summary');
    if (!summary) return;

    if (!study || !experiment) {
        summary.innerHTML = '<p class="planner-summary-empty">No experiment selected.</p>';
        return;
    }

    const studyLabel = study.label || '-';
    const experimentLabel = experiment.label || '-';
    const studyDescription = study.description ? String(study.description) : '';
    const paramsRowsHtml = buildParamsRowsHtml(study.fixed_params, experiment.params);

    summary.innerHTML =
        '<div class="planner-summary-header">' +
            '<div class="planner-summary-chip">' +
                '<span class="planner-summary-label">Study Group</span>' +
                '<strong>' + escapeHtml(studyLabel) + '</strong>' +
            '</div>' +
            '<div class="planner-summary-chip">' +
                '<span class="planner-summary-label">Experiment</span>' +
                '<strong>' + escapeHtml(experimentLabel) + '</strong>' +
            '</div>' +
        '</div>' +
        (studyDescription ? '<p class="planner-summary-description">' + escapeHtml(studyDescription) + '</p>' : '') +
        '<table class="planner-params-table" aria-label="Experiment parameter summary">' +
            '<thead>' +
                '<tr>' +
                    '<th>Parameter</th>' +
                    '<th>Value</th>' +
                    '<th>Scope</th>' +
                '</tr>' +
            '</thead>' +
            '<tbody>' + paramsRowsHtml + '</tbody>' +
        '</table>';
}

function ensurePlannerSelectors(manifest) {
    const studySelect = document.getElementById('planner-study-select');
    const experimentSelect = document.getElementById('planner-experiment-select');
    if (!studySelect || !experimentSelect) return;

    const keys = studyKeys(manifest);
    if (keys.length === 0) return;

    const previousStudyValue = studySelect.value;
    const previousExperimentValue = experimentSelect.value;

    studySelect.innerHTML = '';
    keys.forEach(function(key) {
        const label = manifest.studies[key].label || key;
        appendSelectOption(studySelect, key, label);
    });

    let selectedStudyKey = (previousStudyValue && keys.indexOf(previousStudyValue) !== -1) ? previousStudyValue : '';
    if (!selectedStudyKey) {
        selectedStudyKey = (manifest.active_study && keys.indexOf(manifest.active_study) !== -1) ? manifest.active_study : keys[0];
    }
    studySelect.value = selectedStudyKey;

    const selectedStudy = manifest.studies[selectedStudyKey];
    const experiments = visibleExperiments(selectedStudy);
    experimentSelect.innerHTML = '';

    experiments.forEach(function(experiment) {
        appendSelectOption(experimentSelect, experiment.id, experiment.label);
    });

    let selectedExperimentId = '';
    const availableIds = experiments.map(function(experiment) { return experiment.id; });
    if (previousExperimentValue && availableIds.indexOf(previousExperimentValue) !== -1) {
        selectedExperimentId = previousExperimentValue;
    } else if (manifest.active_experiment && availableIds.indexOf(manifest.active_experiment) !== -1) {
        selectedExperimentId = manifest.active_experiment;
    } else if (experiments.length > 0) {
        selectedExperimentId = experiments[0].id;
    }
    experimentSelect.value = selectedExperimentId;

    if (!plannerUiBound) {
        studySelect.addEventListener('change', function() {
            renderPlannerPlots();
        });
        experimentSelect.addEventListener('change', function() {
            renderPlannerPlots();
        });
        plannerUiBound = true;
    }
}

function selectedStudyAndExperiment(manifest) {
    const keys = studyKeys(manifest);
    if (keys.length === 0) {
        throw new Error('Manifest has no studies.');
    }

    const studySelect = document.getElementById('planner-study-select');
    const experimentSelect = document.getElementById('planner-experiment-select');

    let studyKey = (studySelect && studySelect.value && keys.indexOf(studySelect.value) !== -1)
        ? studySelect.value
        : ((manifest.active_study && keys.indexOf(manifest.active_study) !== -1) ? manifest.active_study : keys[0]);

    const study = manifest.studies[studyKey];
    const experiments = visibleExperiments(study);
    if (experiments.length === 0) {
        throw new Error('Study "' + studyKey + '" has no enabled experiments.');
    }

    const experimentIds = experiments.map(function(experiment) { return experiment.id; });
    let experimentId = (experimentSelect && experimentSelect.value && experimentIds.indexOf(experimentSelect.value) !== -1)
        ? experimentSelect.value
        : ((manifest.active_experiment && experimentIds.indexOf(manifest.active_experiment) !== -1) ? manifest.active_experiment : experiments[0].id);

    const experiment = experiments.find(function(item) {
        return item.id === experimentId;
    }) || experiments[0];

    if (studySelect) studySelect.value = studyKey;
    if (experimentSelect) experimentSelect.value = experiment.id;

    return {
        study_key: studyKey,
        study: study,
        experiment: experiment
    };
}

function trajectoryByName(trajectories, name) {
    return trajectories.find(function(trajectory) {
        return trajectory && trajectory.name === name;
    }) || null;
}

function preferredPathTrajectory(trajectories) {
    return trajectoryByName(trajectories, 'path_replay') || trajectoryByName(trajectories, 'path_estimated');
}

function numeric(value) {
    return typeof value === 'number' && Number.isFinite(value);
}

function buildCloudTrace(points, colorField, title, colorscale) {
    const x = [];
    const y = [];
    const z = [];
    const color = [];
    const normalizedField = colorField + '_norm';

    points.forEach(function(point) {
        const pointX = point && point.x;
        const pointY = point && point.y;
        const pointZ = point && point.z;
        const pointColor = point && (numeric(point[colorField]) ? point[colorField] : point[normalizedField]);

        if (numeric(pointX) && numeric(pointY) && numeric(pointZ) && numeric(pointColor)) {
            x.push(pointX);
            y.push(pointY);
            z.push(pointZ);
            color.push(pointColor);
        }
    });

    return {
        type: 'scatter3d',
        mode: 'markers',
        name: 'Samples',
        x: x,
        y: y,
        z: z,
        marker: {
            size: CLOUD_POINT_SIZE,
            opacity: CLOUD_POINT_OPACITY,
            color: color,
            colorscale: colorscale,
            colorbar: {
                title: {
                    text: title
                }
            }
        },
        hovertemplate: 'x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<br>value=%{marker.color:.4f}<extra></extra>'
    };
}

function buildPathTraces(points) {
    const x = [];
    const y = [];
    const z = [];

    points.forEach(function(point) {
        if (numeric(point.x) && numeric(point.y) && numeric(point.z)) {
            x.push(point.x);
            y.push(point.y);
            z.push(point.z);
        }
    });

    if (x.length === 0) {
        return [];
    }

    const startIndex = 0;
    const endIndex = x.length - 1;
    const zOffset = z.map(function(value) {
        return value + PATH_Z_OFFSET;
    });

    return [
        {
            type: 'scatter3d',
            mode: 'lines',
            name: 'Path outline',
            x: x,
            y: y,
            z: zOffset,
            showlegend: false,
            line: {
                color: 'rgba(15, 23, 42, 0.45)',
                width: PATH_LINE_OUTLINE_WIDTH
            }
        },
        {
            type: 'scatter3d',
            mode: 'lines',
            name: 'Path',
            x: x,
            y: y,
            z: zOffset,
            line: {
                color: '#F28E2B',
                width: PATH_LINE_WIDTH
            }
        },
        {
            type: 'scatter3d',
            mode: 'markers',
            name: 'Start',
            x: [x[startIndex]],
            y: [y[startIndex]],
            z: [zOffset[startIndex]],
            marker: {
                color: '#1D4ED8',
                size: 8,
                symbol: 'square'
            }
        },
        {
            type: 'scatter3d',
            mode: 'markers',
            name: 'End',
            x: [x[endIndex]],
            y: [y[endIndex]],
            z: [zOffset[endIndex]],
            marker: {
                color: '#DC2626',
                size: 8,
                symbol: 'square'
            }
        }
    ];
}

function buildCloudLayout(title) {
    return {
        title: {
            text: title
        },
        template: 'plotly_white',
        height: 440,
        margin: { l: 0, r: 0, b: 0, t: 40 },
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#e2e8f0',
            borderwidth: 1
        },
        scene: {
            xaxis: { title: 'X (m)', gridcolor: '#dbe2ea' },
            yaxis: { title: 'Y (m)', gridcolor: '#dbe2ea' },
            zaxis: { title: 'Z (m)', gridcolor: '#dbe2ea' },
            aspectmode: 'data',
            camera: cameraFromMatplotlib(CAM_ELEV, CAM_AZIM, CAM_ROLL, CAM_DISTANCE)
        }
    };
}

function buildLineTraces(points, keys, colors) {
    return keys.map(function(key, index) {
        const x = [];
        const y = [];

        points.forEach(function(point) {
            if (!point) return;
            if (numeric(point.t) && numeric(point[key])) {
                x.push(point.t);
                y.push(point[key]);
            }
        });

        return {
            type: 'scatter',
            mode: 'lines',
            name: key,
            x: x,
            y: y,
            line: {
                width: 3,
                color: colors[index]
            }
        };
    });
}

function buildLineLayout(title, yLabel) {
    return {
        title: {
            text: title
        },
        template: 'plotly_white',
        height: 440,
        margin: { l: 55, r: 20, b: 55, t: 40 },
        xaxis: {
            title: 'Time [s]',
            gridcolor: '#dbe2ea'
        },
        yaxis: {
            title: yLabel,
            gridcolor: '#dbe2ea'
        },
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#e2e8f0',
            borderwidth: 1
        }
    };
}

async function fetchJson(path, timeoutMs) {
    const timeout = typeof timeoutMs === 'number' ? timeoutMs : 12000;
    const pageUrl = new URL(window.location.href);
    var basePath = pageUrl.pathname;
    var lastSegment = basePath.substring(basePath.lastIndexOf('/') + 1);

    // Normalize base path so relative fetches still work with/without trailing slash.
    if (!basePath.endsWith('/')) {
        if (lastSegment.indexOf('.') !== -1) {
            basePath = basePath.substring(0, basePath.lastIndexOf('/') + 1);
        } else {
            basePath = basePath + '/';
        }
    }

    const baseUrl = pageUrl.origin + basePath;
    const resolvedPath = new URL(path, baseUrl).toString();
    const controller = new AbortController();
    const timer = setTimeout(function() {
        controller.abort();
    }, timeout);

    let response;
    try {
        response = await fetch(resolvedPath, {
            cache: 'no-store',
            signal: controller.signal
        });
    } finally {
        clearTimeout(timer);
    }

    if (!response.ok) {
        throw new Error('Failed to fetch ' + resolvedPath + ' (' + response.status + ')');
    }
    return response.json();
}

async function resolvePlotManifest() {
    try {
        const rawManifest = await fetchJson('data/manifest.json', 10000);
        return normalizePlotManifest(rawManifest);
    } catch (error) {
        // Fallback to default filenames when manifest is not present.
    }
    return normalizeLegacyManifest(DEFAULT_PLOT_DATA_FILES);
}

async function renderPlannerPlots() {
    const requiredIds = ['plot-cost-cloud', 'plot-force-cloud', 'plot-controls', 'plot-velocities'];
    const arePlotsPresent = requiredIds.every(function(id) {
        return !!document.getElementById(id);
    });

    if (!arePlotsPresent) return;

    if (typeof Plotly === 'undefined') {
        setPlannerPlotsStatus('Plotly failed to load. Check network access and refresh.', true);
        return;
    }

    if (window.location.protocol === 'file:') {
        setPlannerPlotsStatus(
            'Open this page through a local server (not file://). Run: python3 -m http.server 8000 and open http://localhost:8000',
            true
        );
        return;
    }

    setPlannerPlotsStatus('Loading plot data...', false);

    try {
        setPlannerPlotsStatus('Loading manifest...', false);
        if (!plannerManifestCache) {
            plannerManifestCache = await resolvePlotManifest();
        }
        ensurePlannerSelectors(plannerManifestCache);
        const selected = selectedStudyAndExperiment(plannerManifestCache);
        const dataSources = selected.experiment;
        setPlannerExperimentSummary(selected.study, selected.experiment);

        setPlannerPlotsStatus('Loading cloud data...', false);
        const cloudData = await fetchJson(dataSources.cloud, 15000);
        setPlannerPlotsStatus('Loading trajectory data...', false);
        const trajectoriesData = await fetchJson(dataSources.trajectories, 15000);

        const cloudPoints = Array.isArray(cloudData.points) ? cloudData.points : [];
        const trajectories = Array.isArray(trajectoriesData.trajectories) ? trajectoriesData.trajectories : [];

        if (cloudPoints.length === 0) {
            throw new Error('Cloud JSON has no points.');
        }
        if (trajectories.length === 0) {
            throw new Error('Trajectory JSON has no trajectories.');
        }

        const pathTrajectory = preferredPathTrajectory(trajectories);
        if (!pathTrajectory || !Array.isArray(pathTrajectory.points) || pathTrajectory.points.length === 0) {
            throw new Error('No valid path trajectory found (expected path_replay or path_estimated).');
        }

        const controlsTrajectory = trajectoryByName(trajectories, 'cables_sent');
        const velocitiesTrajectory = trajectoryByName(trajectories, 'cable_velocities');

        if (!controlsTrajectory || !Array.isArray(controlsTrajectory.points)) {
            throw new Error('No valid cables_sent trajectory found.');
        }
        if (!velocitiesTrajectory || !Array.isArray(velocitiesTrajectory.points)) {
            throw new Error('No valid cable_velocities trajectory found.');
        }

        const pathTraces = buildPathTraces(pathTrajectory.points);

        const costCloudTrace = buildCloudTrace(cloudPoints, 'color_cost', 'Cost', COST_COLOR_SCALE);
        const forceCloudTrace = buildCloudTrace(cloudPoints, 'color_force', 'Force', 'Plasma');
        setPlannerPlotsStatus('Rendering plots...', false);

        await Plotly.newPlot(
            'plot-cost-cloud',
            [costCloudTrace].concat(pathTraces),
            buildCloudLayout('3D Cost Field with Optimal Trajectory'),
            PLOTLY_CONFIG
        );

        await Plotly.newPlot(
            'plot-force-cloud',
            [forceCloudTrace].concat(pathTraces),
            buildCloudLayout('Cable Actuation Cost Cloud'),
            PLOTLY_CONFIG
        );

        const controlTraces = buildLineTraces(controlsTrajectory.points, ['u1', 'u2', 'u3'], ['#1f77b4', '#ff7f0e', '#2ca02c']);
        await Plotly.newPlot(
            'plot-controls',
            controlTraces,
            buildLineLayout('Control Inputs', 'u'),
            PLOTLY_CONFIG
        );

        const velocityTraces = buildLineTraces(velocitiesTrajectory.points, ['v1', 'v2', 'v3'], ['#1f77b4', '#ff7f0e', '#2ca02c']);
        await Plotly.newPlot(
            'plot-velocities',
            velocityTraces,
            buildLineLayout('Control Velocities', 'du/dt'),
            PLOTLY_CONFIG
        );

        setPlannerPlotsStatus('Plots loaded for "' + selected.experiment.label + '".', false);
    } catch (error) {
        console.error(error);
        if (error && error.name === 'AbortError') {
            setPlannerPlotsStatus(
                'Could not load plots: request timeout while reading JSON files. Check GitHub Pages URL and data paths.',
                true
            );
            return;
        }
        if (error && error.name === 'TypeError') {
            setPlannerPlotsStatus(
                'Could not load plots due to a network/fetch error. Make sure the page is served via http://localhost and not opened as a local file.',
                true
            );
            return;
        }
        setPlannerPlotsStatus('Could not load plots: ' + error.message, true);
    }
}

function initPage() {
    // Render plots first so optional UI scripts cannot block this section.
    renderPlannerPlots();

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

    // Initialize carousel only when library is available.
    try {
        if (typeof bulmaCarousel !== 'undefined' && typeof bulmaCarousel.attach === 'function') {
            bulmaCarousel.attach('.carousel', options);
        }

        if (typeof bulmaSlider !== 'undefined' && typeof bulmaSlider.attach === 'function') {
            bulmaSlider.attach();
        }
    } catch (error) {
        console.error('Bulma components failed to initialize.', error);
    }
    
    try {
        // Setup video autoplay for carousel
        setupVideoCarouselAutoplay();
    } catch (error) {
        console.error('Video carousel autoplay setup failed.', error);
    }
}

if (typeof window.jQuery !== 'undefined') {
    window.jQuery(function() {
        initPage();
    });
} else {
    document.addEventListener('DOMContentLoaded', function() {
        initPage();
    });
}
