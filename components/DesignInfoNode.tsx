import React, { memo, useState, useEffect, useRef, useMemo } from 'react';
import { Handle, Position, NodeProps, useEdges, useNodes, Node, NodeResizer } from 'reactflow';
import { SerializableLayer, PSDNodeData, OpticalMetrics } from '../types';
import { useProceduralStore } from '../store/ProceduralContext';
import { findLayerByPath } from '../services/psdService';
import { Layers, Scan, Crosshair, BoxSelect } from 'lucide-react';

interface LayerItemProps {
  node: SerializableLayer;
  depth?: number;
  onHover: (layer: SerializableLayer | null) => void;
  onSelect: (layer: SerializableLayer) => void;
  selectedId: string | null;
}

// Reused Scan Utility for on-demand calculation (if missing from source)
const calculateOpticalMetrics = (canvas: HTMLCanvasElement): OpticalMetrics | null => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    const w = canvas.width;
    const h = canvas.height;
    
    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;
    let minX = w, minY = h, maxX = 0, maxY = 0, found = false;
    let nonTransparentPixels = 0;

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const alpha = data[(y * w + x) * 4 + 3];
            if (alpha > 0) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                found = true;
                nonTransparentPixels++;
            }
        }
    }
    
    const density = (w * h) > 0 ? nonTransparentPixels / (w * h) : 0;
    
    return found ? { 
        bounds: { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 },
        visualCenter: { x: minX + (maxX - minX + 1) / 2, y: minY + (maxY - minY + 1) / 2 },
        pixelDensity: density
    } : null;
};


const LayerItem: React.FC<LayerItemProps> = ({ node, depth = 0, onHover, onSelect, selectedId }) => {
  const [isOpen, setIsOpen] = useState(false);
  const isGroup = node.type === 'group';
  const hasChildren = isGroup && node.children && node.children.length > 0;
  const isSelected = selectedId === node.id;

  const toggleOpen = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (hasChildren) setIsOpen(!isOpen);
  };

  const handleInteraction = (e: React.MouseEvent) => {
      e.stopPropagation();
      onSelect(node);
  };

  return (
    <div className="select-none">
      <div 
        className={`flex items-center py-1 pr-2 rounded cursor-pointer transition-colors ${isSelected ? 'bg-blue-900/40 text-blue-200' : 'hover:bg-slate-700/50 text-slate-400'} ${!node.isVisible ? 'opacity-50' : ''}`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        onClick={handleInteraction}
        onMouseEnter={() => onHover(node)}
        onMouseLeave={() => onHover(null)}
      >
        <div className="mr-1.5 w-4 flex justify-center shrink-0" onClick={toggleOpen}>
          {hasChildren ? (
             <svg 
               className={`w-3 h-3 transition-transform ${isOpen ? 'rotate-90' : ''} ${isSelected ? 'text-blue-400' : 'text-slate-500'}`} 
               fill="none" viewBox="0 0 24 24" stroke="currentColor"
             >
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
             </svg>
          ) : (
            <div className="w-3" />
          )}
        </div>

        <div className="mr-2 shrink-0">
           {isGroup ? (
             <svg className={`w-3.5 h-3.5 ${isSelected ? 'text-blue-300' : 'text-slate-500'}`} fill="currentColor" viewBox="0 0 20 20">
               <path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
             </svg>
           ) : (
             <svg className={`w-3.5 h-3.5 ${isSelected ? 'text-blue-300' : 'text-slate-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
             </svg>
           )}
        </div>

        <span className="text-xs truncate font-medium">{node.name}</span>
        
        {/* Optical Badge if present */}
        {node.optical && (
            <div className="ml-auto flex items-center space-x-1">
                 <span className="w-1.5 h-1.5 rounded-full bg-red-500" title="Optical Metrics Available"></span>
            </div>
        )}
      </div>

      {isOpen && hasChildren && (
        <div className="border-l border-slate-700 ml-[15px]">
          {/* REVERSED: Render top-most children first */}
          {[...node.children!].reverse().map((child) => (
            <LayerItem key={child.id} node={child} depth={depth + 1} onHover={onHover} onSelect={onSelect} selectedId={selectedId} />
          ))}
        </div>
      )}
    </div>
  );
};

// --- VISUAL INSPECTOR COMPONENT ---
const LayerInspector: React.FC<{ 
    layer: SerializableLayer | null, 
    sourcePsdId: string | null 
}> = ({ layer, sourcePsdId }) => {
    const { psdRegistry } = useProceduralStore();
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [localMetrics, setLocalMetrics] = useState<OpticalMetrics | null>(null);
    const [isScanning, setIsScanning] = useState(false);
    
    // Effective Metrics: Persisted (from Analyst) OR Calculated Locally
    const activeMetrics = layer?.optical || localMetrics;

    useEffect(() => {
        if (!layer || !sourcePsdId || !canvasRef.current) {
            setLocalMetrics(null);
            return;
        }

        const psd = psdRegistry[sourcePsdId];
        if (!psd) return;

        // If optical data already exists in the node, use it (skip scan)
        if (layer.optical) {
            setLocalMetrics(null); // Clear local, we use prop
            
            // Just draw the layer for visual context
            const agLayer = findLayerByPath(psd, layer.id);
            if (agLayer && agLayer.canvas) {
                const ctx = canvasRef.current.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                    // Draw centered in view
                    const scale = Math.min(
                        canvasRef.current.width / agLayer.canvas.width, 
                        canvasRef.current.height / agLayer.canvas.height
                    ) * 0.8; // 80% fit
                    
                    const drawW = agLayer.canvas.width * scale;
                    const drawH = agLayer.canvas.height * scale;
                    const drawX = (canvasRef.current.width - drawW) / 2;
                    const drawY = (canvasRef.current.height - drawH) / 2;
                    
                    ctx.drawImage(agLayer.canvas, drawX, drawY, drawW, drawH);
                }
            }
            return;
        }

        // --- PERFORM LOCAL SCAN ---
        setIsScanning(true);
        const agLayer = findLayerByPath(psd, layer.id);
        
        if (agLayer && agLayer.canvas) {
            const canvas = agLayer.canvas as HTMLCanvasElement;
            const metrics = calculateOpticalMetrics(canvas);
            setLocalMetrics(metrics);
            
            // Draw to Inspector Canvas
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                
                // Calculate Fit Scale
                const viewW = canvasRef.current.width;
                const viewH = canvasRef.current.height;
                const scale = Math.min(viewW / canvas.width, viewH / canvas.height) * 0.8;
                
                const drawW = canvas.width * scale;
                const drawH = canvas.height * scale;
                const drawX = (viewW - drawW) / 2;
                const drawY = (viewH - drawH) / 2;
                
                ctx.drawImage(canvas, drawX, drawY, drawW, drawH);
            }
        } else {
            setLocalMetrics(null);
            const ctx = canvasRef.current.getContext('2d');
            ctx?.clearRect(0,0, canvasRef.current.width, canvasRef.current.height);
        }
        setIsScanning(false);

    }, [layer, sourcePsdId, psdRegistry]);

    if (!layer) return (
        <div className="h-40 bg-slate-900 border-b border-slate-700 flex flex-col items-center justify-center text-slate-500 space-y-2">
            <Layers className="w-8 h-8 opacity-20" />
            <span className="text-[10px] uppercase tracking-widest opacity-50">Select Layer to Inspect</span>
        </div>
    );

    // Calculate overlay positions relative to the inspector canvas
    // We need to replicate the scale logic used in the effect above
    const renderScale = layer.coords.w > 0 && layer.coords.h > 0 
        ? Math.min(240 / layer.coords.w, 160 / layer.coords.h) * 0.8 
        : 1;
        
    const offsetX = (240 - (layer.coords.w * renderScale)) / 2;
    const offsetY = (160 - (layer.coords.h * renderScale)) / 2;

    const density = activeMetrics?.pixelDensity ?? 0;
    const densityLabel = density > 0.8 ? 'Solid' : density > 0.3 ? 'Standard' : density > 0.05 ? 'Sparse' : 'Ghost';
    const densityColor = density > 0.8 ? 'text-emerald-400' : density > 0.3 ? 'text-blue-400' : 'text-orange-400';

    return (
        <div className="relative h-40 bg-slate-900 border-b border-slate-700 overflow-hidden group">
            {/* Background Grid */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:20px_20px]"></div>
            
            <canvas 
                ref={canvasRef} 
                width={240} 
                height={160} 
                className="absolute inset-0 w-full h-full z-10"
            />
            
            {/* Overlays */}
            <div className="absolute inset-0 z-20 pointer-events-none">
                 {/* Geometric Bounds (Blue) */}
                 <div 
                    className="absolute border border-blue-500/50"
                    style={{
                        left: offsetX,
                        top: offsetY,
                        width: layer.coords.w * renderScale,
                        height: layer.coords.h * renderScale
                    }}
                 >
                    <span className="absolute -top-3 left-0 text-[7px] text-blue-500 font-mono bg-slate-900/80 px-1">GEOMETRIC</span>
                 </div>

                 {/* Optical Bounds (Red Dashed) */}
                 {activeMetrics && (
                     <div 
                        className="absolute border border-red-500 border-dashed shadow-[0_0_10px_rgba(239,68,68,0.3)]"
                        style={{
                            left: offsetX + (activeMetrics.bounds.x * renderScale),
                            top: offsetY + (activeMetrics.bounds.y * renderScale),
                            width: activeMetrics.bounds.w * renderScale,
                            height: activeMetrics.bounds.h * renderScale
                        }}
                     >
                        <span className="absolute -bottom-3 right-0 text-[7px] text-red-500 font-mono bg-slate-900/80 px-1">OPTICAL</span>
                        {/* Visual Center Crosshair */}
                        {activeMetrics.visualCenter && (
                            <div 
                                className="absolute w-2 h-2 -ml-1 -mt-1 flex items-center justify-center text-red-400"
                                style={{
                                    left: (activeMetrics.visualCenter.x - activeMetrics.bounds.x) * renderScale,
                                    top: (activeMetrics.visualCenter.y - activeMetrics.bounds.y) * renderScale,
                                }}
                            >
                                <Crosshair className="w-3 h-3" />
                            </div>
                        )}
                     </div>
                 )}
            </div>

            {/* HUD / Info Panel */}
            <div className="absolute bottom-0 left-0 right-0 bg-slate-950/80 backdrop-blur-sm p-2 border-t border-slate-700 flex justify-between items-center z-30">
                <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-slate-200 truncate max-w-[120px]">{layer.name}</span>
                    <span className="text-[8px] text-slate-500 font-mono">{Math.round(layer.coords.w)}x{Math.round(layer.coords.h)}px</span>
                </div>
                
                {activeMetrics ? (
                    <div className="flex items-center space-x-3 text-right">
                         <div className="flex flex-col items-end">
                             <span className="text-[8px] uppercase text-slate-500 font-bold tracking-wider">Density</span>
                             <span className={`text-[9px] font-mono ${densityColor}`}>{densityLabel} ({Math.round(density * 100)}%)</span>
                         </div>
                         <div className="flex flex-col items-end">
                            <span className="text-[8px] uppercase text-slate-500 font-bold tracking-wider">Trim</span>
                            <span className="text-[9px] font-mono text-slate-300">
                                {Math.round((1 - (activeMetrics.bounds.w * activeMetrics.bounds.h) / (layer.coords.w * layer.coords.h)) * 100)}%
                            </span>
                         </div>
                    </div>
                ) : (
                    <div className="flex items-center space-x-1 text-slate-500">
                        {isScanning ? <Scan className="w-3 h-3 animate-spin" /> : <BoxSelect className="w-3 h-3" />}
                        <span className="text-[8px] uppercase">{isScanning ? 'Scanning...' : 'No Optics'}</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export const DesignInfoNode = memo(({ id }: NodeProps) => {
  const edges = useEdges();
  const nodes = useNodes();
  const [hoveredLayer, setHoveredLayer] = useState<SerializableLayer | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<SerializableLayer | null>(null);
  
  // Find the source node connected to this node's handle
  const sourceNode = React.useMemo(() => {
    const edge = edges.find(e => e.target === id);
    if (!edge) return null;
    return nodes.find(n => n.id === edge.source) as Node<PSDNodeData> | undefined;
  }, [edges, nodes, id]);

  const designLayers = sourceNode?.data?.designLayers;
  const activeLayer = hoveredLayer || selectedLayer;

  return (
    <div className="w-64 bg-slate-800 rounded-lg shadow-xl border border-slate-600 font-sans flex flex-col overflow-hidden transition-all duration-300 hover:border-blue-500/50">
      <NodeResizer minWidth={256} minHeight={300} isVisible={true} handleStyle={{ background: 'transparent', border: 'none' }} lineStyle={{ border: 'none' }} />

      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !top-8 !bg-blue-500 !border-2 !border-slate-800"
        title="Input"
      />

      {/* Header Container */}
      <div className="bg-slate-900 p-2 border-b border-slate-700 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-2">
          <div className="p-1 bg-blue-500/20 rounded border border-blue-500/30">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
          </div>
          <div className="flex flex-col leading-none">
             <span className="text-sm font-semibold text-slate-200">Design Info</span>
             <span className="text-[9px] text-blue-400 font-mono">HIERARCHY</span>
          </div>
        </div>
      </div>

      {/* VISUAL INSPECTION PANE */}
      <LayerInspector layer={activeLayer} sourcePsdId={sourceNode?.id || null} />

      {/* Content */}
      <div className="flex-1 bg-slate-800 overflow-y-auto custom-scrollbar p-1 min-h-[150px]">
        {!sourceNode ? (
          <div className="flex flex-col items-center justify-center h-24 text-slate-500 p-4 text-center">
            <svg className="w-8 h-8 mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
            <span className="text-xs">Connect a Loaded PSD Node</span>
          </div>
        ) : !designLayers ? (
          <div className="flex flex-col items-center justify-center h-24 text-slate-500 text-xs">
            <span>No design layers found.</span>
          </div>
        ) : (
          <div className="py-1">
             {/* REVERSED: Render top-most layers first (Photoshop Style) */}
             {[...designLayers].reverse().map(layer => (
               <LayerItem 
                   key={layer.id} 
                   node={layer} 
                   onHover={setHoveredLayer} 
                   onSelect={setSelectedLayer}
                   selectedId={selectedLayer?.id || null}
               />
             ))}
          </div>
        )}
      </div>
    </div>
  );
});