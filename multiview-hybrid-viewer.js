const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3005;

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'web', 'multiview-hybrid-viewer.html'));
});

app.get('/api/folders', (req, res) => {
  const outputDir = path.join(__dirname, 'output');

  fs.readdir(outputDir, { withFileTypes: true }, (err, entries) => {
    if (err) {
      res.status(500).json({ error: 'Cannot read output directory' });
      return;
    }

    const folders = entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort((a, b) => {
        const aTime = fs.statSync(path.join(outputDir, a)).mtimeMs;
        const bTime = fs.statSync(path.join(outputDir, b)).mtimeMs;
        return bTime - aTime;
      });

    res.json({ folders });
  });
});

app.get('/api/meshes/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(__dirname, 'output', folder);

  const resolvedPath = path.resolve(folderPath);
  const outputBase = path.resolve(path.join(__dirname, 'output'));
  if (!resolvedPath.startsWith(outputBase)) {
    res.status(403).json({ error: 'Access denied' });
    return;
  }

  if (!fs.existsSync(folderPath)) {
    res.status(404).json({ error: 'Folder not found' });
    return;
  }

  fs.readdir(folderPath, (err, files) => {
    if (err) {
      res.status(500).json({ error: 'Cannot read folder' });
      return;
    }

    const meshFiles = files
      .filter((file) => {
        const lower = file.toLowerCase();
        return lower.endsWith('.ply') || lower.endsWith('.glb');
      })
      .sort((a, b) => {
        const aTime = fs.statSync(path.join(folderPath, a)).mtimeMs;
        const bTime = fs.statSync(path.join(folderPath, b)).mtimeMs;
        return bTime - aTime;
      });

    res.json({ folder, meshes: meshFiles });
  });
});

function getCamIndexFromName(name) {
  const match = String(name || '').match(/cam(\d+)/i);
  if (!match) return null;
  const idx = Number.parseInt(match[1], 10);
  return Number.isFinite(idx) ? idx : null;
}

function inferSplatBaseFromMeshFile(meshName) {
  const stem = meshName.replace(/\.(ply|glb)$/i, '');
  const match = stem.match(/^(.*)_cam\d+/i);
  if (match && match[1]) return match[1];
  return (stem.split('_')[0] || stem).trim();
}

function inferCommonBase(meshFiles) {
  const counts = new Map();
  for (const meshFile of meshFiles) {
    const stem = meshFile.replace(/\.(ply|glb)$/i, '');
    const match = stem.match(/^(.*)_cam\d+/i);
    const base = match && match[1] ? match[1] : stem;
    if (!base) continue;
    counts.set(base, (counts.get(base) || 0) + 1);
  }
  let best = '';
  let bestCount = 0;
  counts.forEach((count, base) => {
    if (count > bestCount) {
      bestCount = count;
      best = base;
    }
  });
  if (best) return best;
  return meshFiles.length ? inferSplatBaseFromMeshFile(meshFiles[0]) : '';
}

function buildCameraTxtCandidates(meshFile) {
  const stem = meshFile.replace(/\.(ply|glb)$/i, '');
  const base = stem.replace(/_mesh_multi$/i, '').replace(/_mesh$/i, '');
  const candidates = [`${stem}_camera_main_view.txt`];
  if (base !== stem) {
    candidates.push(`${base}_camera_main_view.txt`);
  }
  return candidates;
}

function parseNamed4x4Block(text, blockName) {
  const escaped = blockName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`${escaped}:\\s*\\n([^\\n]+)\\n([^\\n]+)\\n([^\\n]+)\\n([^\\n]+)`, 'i');
  const match = text.match(re);
  if (!match) return null;

  const rows = [match[1], match[2], match[3], match[4]];
  const nums = rows.join(' ').trim().split(/[\s,]+/).map(Number).filter(Number.isFinite);
  if (nums.length !== 16) return null;
  return nums;
}

function readPlyHeader(buffer) {
  const endIdx = buffer.indexOf('end_header');
  if (endIdx === -1) return null;
  let headerEnd = buffer.indexOf('\n', endIdx);
  if (headerEnd === -1) {
    headerEnd = endIdx + Buffer.byteLength('end_header');
  } else {
    headerEnd += 1;
  }

  const headerText = buffer.slice(0, headerEnd).toString('utf8');
  const lines = headerText.split(/\r?\n/).filter(Boolean);

  let format = null;
  let vertexCount = 0;
  let inVertex = false;
  const properties = [];

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (!parts.length) continue;
    if (parts[0] === 'format' && parts[1]) {
      format = parts[1];
    } else if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
      if (inVertex) {
        vertexCount = Number.parseInt(parts[2], 10) || 0;
      }
    } else if (inVertex && parts[0] === 'property' && parts.length >= 3) {
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  return { format, vertexCount, properties, headerEnd };
}

function plyTypeInfo(type) {
  const map = {
    char: { size: 1, method: 'getInt8' },
    int8: { size: 1, method: 'getInt8' },
    uchar: { size: 1, method: 'getUint8' },
    uint8: { size: 1, method: 'getUint8' },
    short: { size: 2, method: 'getInt16' },
    int16: { size: 2, method: 'getInt16' },
    ushort: { size: 2, method: 'getUint16' },
    uint16: { size: 2, method: 'getUint16' },
    int: { size: 4, method: 'getInt32' },
    int32: { size: 4, method: 'getInt32' },
    uint: { size: 4, method: 'getUint32' },
    uint32: { size: 4, method: 'getUint32' },
    float: { size: 4, method: 'getFloat32' },
    float32: { size: 4, method: 'getFloat32' },
    double: { size: 8, method: 'getFloat64' },
    float64: { size: 8, method: 'getFloat64' },
  };
  return map[type] || null;
}

function median(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function computeMedianPositionFromPly(filePath) {
  let buffer = null;
  try {
    buffer = fs.readFileSync(filePath);
  } catch {
    return null;
  }

  const header = readPlyHeader(buffer);
  if (!header || !header.vertexCount || !header.format) return null;

  const xIndex = header.properties.findIndex((p) => p.name === 'x');
  const yIndex = header.properties.findIndex((p) => p.name === 'y');
  const zIndex = header.properties.findIndex((p) => p.name === 'z');
  if (xIndex < 0 || yIndex < 0 || zIndex < 0) return null;

  const typeInfo = header.properties.map((p) => plyTypeInfo(p.type));
  if (typeInfo.some((t) => !t)) return null;

  const xVals = [];
  const yVals = [];
  const zVals = [];

  if (header.format === 'ascii') {
    const body = buffer.toString('utf8', header.headerEnd);
    const lines = body.split(/\r?\n/).filter(Boolean);
    for (let i = 0; i < Math.min(lines.length, header.vertexCount); i++) {
      const parts = lines[i].trim().split(/\s+/).map(Number);
      const x = parts[xIndex];
      const y = parts[yIndex];
      const z = parts[zIndex];
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        xVals.push(x);
        yVals.push(y);
        zVals.push(z);
      }
    }
  } else if (header.format === 'binary_little_endian') {
    const dv = new DataView(buffer.buffer, buffer.byteOffset + header.headerEnd);
    const stride = typeInfo.reduce((acc, info) => acc + info.size, 0);
    const getValue = (offset, info) => dv[info.method](offset, true);

    for (let i = 0; i < header.vertexCount; i++) {
      let offset = i * stride;
      let x = null;
      let y = null;
      let z = null;

      for (let p = 0; p < typeInfo.length; p++) {
        const info = typeInfo[p];
        const value = getValue(offset, info);
        if (p === xIndex) x = value;
        if (p === yIndex) y = value;
        if (p === zIndex) z = value;
        offset += info.size;
      }

      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        xVals.push(x);
        yVals.push(y);
        zVals.push(z);
      }
    }
  } else {
    return null;
  }

  const mx = median(xVals);
  const my = median(yVals);
  const mz = median(zVals);
  if (!Number.isFinite(mx) || !Number.isFinite(my) || !Number.isFinite(mz)) return null;
  return [mx, my, mz];
}

function invertRigidW2C(w2c) {
  const r00 = w2c[0], r01 = w2c[1], r02 = w2c[2], tx = w2c[3];
  const r10 = w2c[4], r11 = w2c[5], r12 = w2c[6], ty = w2c[7];
  const r20 = w2c[8], r21 = w2c[9], r22 = w2c[10], tz = w2c[11];

  const rt00 = r00, rt01 = r10, rt02 = r20;
  const rt10 = r01, rt11 = r11, rt12 = r21;
  const rt20 = r02, rt21 = r12, rt22 = r22;

  const t0 = -(rt00 * tx + rt01 * ty + rt02 * tz);
  const t1 = -(rt10 * tx + rt11 * ty + rt12 * tz);
  const t2 = -(rt20 * tx + rt21 * ty + rt22 * tz);

  return [
    rt00, rt01, rt02, t0,
    rt10, rt11, rt12, t1,
    rt20, rt21, rt22, t2,
    0, 0, 0, 1,
  ];
}

function cameraFromW2C(w2c) {
  const c2w = invertRigidW2C(w2c);
  const pos = [c2w[3], c2w[7], c2w[11]];
  const fwd = [c2w[2], c2w[6], c2w[10]]; // camera +Z in world
  return { position: pos, forward: fwd };
}

app.get('/api/cameras/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(__dirname, 'output', folder);

  const resolvedPath = path.resolve(folderPath);
  const outputBase = path.resolve(path.join(__dirname, 'output'));
  if (!resolvedPath.startsWith(outputBase)) {
    res.status(403).json({ error: 'Access denied' });
    return;
  }

  if (!fs.existsSync(folderPath)) {
    res.status(404).json({ error: 'Folder not found' });
    return;
  }

  fs.readdir(folderPath, (err, files) => {
    if (err) {
      res.status(500).json({ error: 'Cannot read folder' });
      return;
    }

    const meshFiles = files.filter((file) => {
      const lower = file.toLowerCase();
      return lower.endsWith('.ply') || lower.endsWith('.glb');
    });

    const cameras = [];
    for (const meshFile of meshFiles) {
      const camIndex = getCamIndexFromName(meshFile);
      if (camIndex === null) continue;

      const candidates = buildCameraTxtCandidates(meshFile);
      let cameraFile = null;
      for (const candidate of candidates) {
        const candidatePath = path.join(folderPath, candidate);
        if (fs.existsSync(candidatePath)) {
          cameraFile = candidate;
          break;
        }
      }

      if (!cameraFile) continue;

      const cameraPath = path.join(folderPath, cameraFile);
      let content = null;
      try {
        content = fs.readFileSync(cameraPath, 'utf8');
      } catch {
        continue;
      }

      let w2c = parseNamed4x4Block(content, 'extrinsic_world_to_camera_4x4');
      if (!w2c) {
        const c2w = parseNamed4x4Block(content, 'extrinsic_camera_to_world_4x4');
        if (c2w) {
          w2c = invertRigidW2C(c2w);
        }
      }

      if (!w2c) continue;

      const { position, forward } = cameraFromW2C(w2c);
      cameras.push({
        id: camIndex,
        camIndex,
        meshFile,
        cameraFile,
        position,
        forward,
      });
    }

    let objectCenter = null;
    let objectCenterSource = null;
    if (meshFiles.length) {
      const base = inferCommonBase(meshFiles);
      const candidateInOutput = path.join(folderPath, `${base}_visible_splats.ply`);
      const candidateInSplats = path.join(__dirname, 'splats', `${base}.ply`);

      const candidates = [];
      if (base) {
        candidates.push(candidateInSplats, candidateInOutput);
      }

      if (!candidates.length) {
        const outputCandidates = files
          .filter((f) => f.endsWith('_visible_splats.ply'))
          .map((f) => path.join(folderPath, f));
        candidates.push(...outputCandidates);
      }

      for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
          const center = computeMedianPositionFromPly(candidate);
          if (center) {
            objectCenter = center;
            objectCenterSource = candidate;
            break;
          }
        }
      }

      if (!objectCenter) {
        try {
          const splatDir = path.join(__dirname, 'splats');
          const splatFiles = fs.readdirSync(splatDir).filter((f) => f.toLowerCase().endsWith('.ply'));
          if (splatFiles.length === 1) {
            const only = path.join(splatDir, splatFiles[0]);
            const center = computeMedianPositionFromPly(only);
            if (center) {
              objectCenter = center;
              objectCenterSource = only;
            }
          }
        } catch {
          // optional fallback
        }
      }
    }

    res.json({ folder, cameras, object_center: objectCenter, object_center_source: objectCenterSource });
  });
});

app.use('/output', express.static(path.join(__dirname, 'output')));
app.use('/splats', express.static(path.join(__dirname, 'splats')));

app.use((req, res) => {
  res.status(404).send('404 Not Found');
});

app.listen(PORT, () => {
  console.log(`Multi-view hybrid viewer server running at http://localhost:${PORT}/`);
});
