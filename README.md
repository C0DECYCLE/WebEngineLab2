# 🪶 WebEngine Lab 2

> Experimental lab 2 for the lightweight fast graphics engine for the web.

-   [x] dependency injection, import or provider?
-   [ ] clean code robert c martin
-   [x] write tests, fix import extension
-   [x] clean up webgpu, do recap
-   [ ] make small lightweight engine/renderer
-   [ ] make planets (chunk system)
-   [ ] entity graph
-   [ ] astroneer look
-   [ ] cloth simulation
-   [x] scattering system 100% gpu
-   [ ] object container streaming
-   [x] gpu folliage
-   [ ] ghost of tsushima techniques (https://www.youtube.com/results?search_query=ghost+of+tsushima+gdc)
-   [x] gpu grass
-   [ ] screen space gpu particles
-   [ ] volumetric clouds
-   [ ] screen space shadows (https://panoskarabelas.com/posts/screen_space_shadows/)
-   [ ] astroid cluster, ring, render haze around ring
-   [ ] look at star citizen, ghost of tsushima and space repo for features
-   [ ] webgpu offscreen canvas
-   [ ] worker multi threading
-   [x] compute shader
-   [x] deferred shading
-   [ ] gpu quadtree, quadtree compute shader
-   [x] grass shading phong & color along blade, curved normals?
-   [x] grass turn around randomly with tilt towards camera
-   [x] grass animate with wind field, random grass scale,
-   [x] grass perlin distribution with clumps, look for other basics then, optimization
-   [ ] gpu floating origin https://godotengine.org/article/emulating-double-precision-gpu-render-large-worlds/
-   [x] drawindircet https://developer.mozilla.org/en-US/docs/Web/API/GPURenderPassEncoder/drawIndirect
-   [x] gpu frustum culling
-   [x] gpu timing
-   [x] https://math.hws.edu/graphicsbook/c9/s2.html
-   [x] indexed drawing
-   [x] bézier curve and smooth normals generated in vertex

//miniature 3d stylized witcher like rpg procedural prototype
//depth of field, advanced lighting (deferred?, global illumination)
//fog, procedural terrain, procedural folliage vegetation grass
//monsters, main character, loot, witcher like
//webgpu, prototype, low poyl? flat smooth or toon shading? normals?
//procedural towns buildings
//inspiration

//frustum cull compute
//just basic flat terrain, color it
//spawn vetegetation

# Virtual Geometry

## Resources

### Nanite

-   https://www.youtube.com/watch?v=eviSykqSUUw
-   https://www.youtube.com/watch?v=NRnj_lnpORU&t=5027s
-   https://www.youtube.com/watch?v=TMorJX3Nj6U&t=3982s

-   https://cs418.cs.illinois.edu/website/text/nanite.html
-   https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf
-   https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf
-   https://computergraphics.stackexchange.com/questions/11060/what-are-mesh-clusters-hierarchical-cluster-culling-with-lod-triangle-c
-   https://www.elopezr.com/a-macro-view-of-nanite/
-   https://www.notion.so/Brief-Analysis-of-Nanite-94be60f292434ba3ae62fa4bcf7d9379
-   https://gamedev.stackexchange.com/questions/198454/how-does-unreal-engine-5s-nanite-work

### Other

-   https://discourse.threejs.org/t/virtually-geometric/28420/40

## Idea

### Instance

position info
bounding info
id unique globaly

### Cluster

max 128 tris indexed
boudning info
id unique for instance, position in linear? (to find parent, siblings, children?)
store tree in one buffer and all the cluster data in another? or same

### Compiletime

import and parse geometry
divde into clusters, merge upwards and simplify and build tree
(binary, quad or dag) encode in linear format
build attributes for all clusters
for each instance build its info (only compile or also run?)

### Runtime

(first sync cpu-gpu object changes?)
for each instance cull in compute (reduce as much as possible) (go trough instance buffer and write survived ids into new buffer)
for each surviving instance and its geometry tree determine the lod level and cull (reduce as much as possible)
add all still remaining clusters (aka its instance id and cluster id) to a (draw) buffer
issue one non indexed draw call on that buffer (vertexCount = 128 tris \* 3, instanceCount = number of clusters)
and the draw shader fetches the correct data via the instance_index and vertex_index:
use the instance_index to find the current cluster data (instanceId of object and clusterId inside object)
then u can get the object position via the instanceId into the instance buffer
and you can get the index of the vertex via the clusterId + vertex_index into the index buffer which will point to the vertex buffer

### Notes

while import check for limits, max number of indexes (half of 32bit ? because of)
use float16 somewhere?
do all dispatchWorkgroupIndirect amazing!!!
build softare rasterizer? for small clusters? triangles?
build occlusing culling? two pass hirarchical z buffer
build non lod cracks? dag graph partitiong
multiple shading modes for debug
check which lod to show by calculating its screen size (use bounding sphere info to project to points onto screen)

every update: record all object property changes and write them chunk vise all together before draw process, every object holds its index in the instance buffer, if deleted the place gets registered in the cpu as free and id a new gets created it gets filled, this would mean holes🤔 not bad because compute shader can skip them? or should they be filled up by swapping last one in? anyway egal which case it requires a cpu-gpu memory write. how do they work around buffer limits? wouldnt that mean a general instance limit?
