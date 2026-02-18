const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3001;

// Serve index2.html at root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'web', 'index2.html'));
});

// API endpoint to list folders in ./output/
app.get('/api/folders', (req, res) => {
  const outputDir = path.join(__dirname, 'output');
  
  fs.readdir(outputDir, { withFileTypes: true }, (err, entries) => {
    if (err) {
      res.status(500).json({ error: 'Cannot read output directory' });
      return;
    }
    
    // Filter only directories
    const folders = entries
      .filter(entry => entry.isDirectory())
      .map(entry => entry.name)
      .sort((a, b) => {
        const aTime = fs.statSync(path.join(outputDir, a)).mtimeMs;
        const bTime = fs.statSync(path.join(outputDir, b)).mtimeMs;
        return bTime - aTime; // Most recent first
      });
    
    res.json({ folders });
  });
});

// API endpoint to list GS and mesh files in a specific folder
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
    
    // Filter for _pruned.ply, _gs.ply (GS) and _mesh.ply (mesh) files
    const gsFiles = files.filter(f => f.endsWith('_pruned.ply') || f.endsWith('_gs.ply'));
    const meshFiles = files.filter(f => f.endsWith('_mesh.ply'));
    
    // Also check for corresponding .json files for metadata
    const filesWithMeta = [];
    
    [...gsFiles, ...meshFiles].forEach(file => {
      const baseName = file.replace(/\.ply$/, '');
      const jsonFile = baseName + '.json';
      const jsonPath = path.join(folderPath, jsonFile);
      
      let meta = null;
      if (fs.existsSync(jsonPath)) {
        try {
          meta = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        } catch (e) {
          // Ignore JSON parse errors
        }
      }
      
      filesWithMeta.push({
        name: file,
        type: (file.endsWith('_pruned.ply') || file.endsWith('_gs.ply')) ? 'gs' : 'mesh',
        meta: meta
      });
    });
    
    res.json({ 
      folder,
      files: filesWithMeta 
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
  console.log(`Web2 server running at http://localhost:${PORT}/`);
});
