const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3004;

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'web', 'hybrid-viewer.html'));
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

app.use('/output', express.static(path.join(__dirname, 'output')));
app.use('/splats', express.static(path.join(__dirname, 'splats')));

app.use((req, res) => {
  res.status(404).send('404 Not Found');
});

app.listen(PORT, () => {
  console.log(`Hybrid viewer server running at http://localhost:${PORT}/`);
});
