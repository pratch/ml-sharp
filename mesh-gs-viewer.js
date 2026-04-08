const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3003;

function readPlyHeader(filePath, maxBytes = 64 * 1024) {
  let fd;
  try {
    fd = fs.openSync(filePath, 'r');
    const buffer = Buffer.alloc(maxBytes);
    const bytesRead = fs.readSync(fd, buffer, 0, maxBytes, 0);
    const content = buffer.toString('utf8', 0, bytesRead);

    const endHeaderIndex = content.indexOf('end_header');
    if (endHeaderIndex === -1) {
      return null;
    }

    return content.slice(0, endHeaderIndex + 'end_header'.length);
  } catch (error) {
    return null;
  } finally {
    if (fd !== undefined) {
      try {
        fs.closeSync(fd);
      } catch (error) {
        // Ignore close errors.
      }
    }
  }
}

function detectPlyType(filePath) {
  const lowerName = path.basename(filePath).toLowerCase();
  if (lowerName.endsWith('.glb')) {
    return 'mesh';
  }
  const header = readPlyHeader(filePath);

  if (!header) {
    if (lowerName.endsWith('_pruned.ply') || lowerName.endsWith('_gs.ply')) {
      return 'gs';
    }
    return 'mesh';
  }

  const hasFaceElement = /\belement\s+face\s+([1-9]\d*)\b/i.test(header);
  if (hasFaceElement) {
    return 'mesh';
  }

  // Typical 3D Gaussian Splat properties used in PLY exports.
  const hasGaussianProperty = /\bproperty\s+\w+\s+(scale_[0-2]|rot_[0-3]|opacity|f_dc_\d+|f_rest_\d+)\b/i.test(header);
  if (hasGaussianProperty) {
    return 'gs';
  }

  if (lowerName.endsWith('_pruned.ply') || lowerName.endsWith('_gs.ply')) {
    return 'gs';
  }

  return 'mesh';
}

// Serve index2.html at root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'web', 'mesh-gs-viewer.html'));
});

// API endpoint to list folders in ./output/
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

// API endpoint to list all PLY files in a specific folder and classify each as mesh or GS
app.get('/api/files/:folder', (req, res) => {
  const folder = req.params.folder;
  const folderPath = path.join(__dirname, 'output', folder);

  // Security check: prevent directory traversal
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

    const plyFiles = files.filter((file) => file.toLowerCase().endsWith('.ply') || file.toLowerCase().endsWith('.glb'));

    const filesWithMeta = plyFiles.map((file) => {
      const filePath = path.join(folderPath, file);
      const baseName = file.replace(/\.(ply|glb)$/i, '');
      const jsonFile = `${baseName}.json`;
      const jsonPath = path.join(folderPath, jsonFile);

      let meta = null;
      if (fs.existsSync(jsonPath)) {
        try {
          meta = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        } catch (error) {
          // Ignore JSON parse errors.
        }
      }

      return {
        name: file,
        type: detectPlyType(filePath),
        meta,
      };
    });

    res.json({
      folder,
      files: filesWithMeta,
    });
  });
});

// Serve static files from output directory
app.use('/output', express.static(path.join(__dirname, 'output')));

// 404 handler
app.use((req, res) => {
  res.status(404).send('404 Not Found');
});

app.listen(PORT, () => {
  console.log(`Mesh/GS viewer server running at http://localhost:${PORT}/`);
});
