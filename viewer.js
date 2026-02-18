const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3002;

// Serve viewer.html at root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'web', 'viewer.html'));
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

// API endpoint to list mesh files in a specific folder
app.get('/api/meshes/:folder', (req, res) => {
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
    
    // Filter for _mesh.ply files
    const meshFiles = files.filter(f => f.endsWith('_mesh.ply'));
    
    res.json({ 
      folder,
      meshes: meshFiles 
    });
  });
});

// Serve static files from output directory
app.use('/output', express.static(path.join(__dirname, 'output')));

// Serve static files from splats directory
app.use('/splats', express.static(path.join(__dirname, 'splats')));

// 404 handler
app.use((req, res) => {
  res.status(404).send('404 Not Found');
});

app.listen(PORT, () => {
  console.log(`Viewer server running at http://localhost:${PORT}/`);
});
