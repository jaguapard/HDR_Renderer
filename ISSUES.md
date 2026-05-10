# General

- Scene and renderer state switching machine is duct-taped spaghetti
- Heavy dependence on AVX512, intrinsics everywhere
- Renderers too aware of the scene, gotta separate it into a class

# Rasterizing renderer
- Some transparent pixels are sky-colored, most notable during scene explosion
- No support for different projections, homogenious space unused
- Multitexturing crashes on Clang but not MSVC (outdated and deprecated concept anyway)


# Ray casting renderer
- Ray casting renderer can't launch New Sponza scene: crashes with all octree nodes refusing some triangle
- Octree structure and quality are very mediocre
- Shadow rays kill performance at unlucky angles