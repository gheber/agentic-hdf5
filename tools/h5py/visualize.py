import h5py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import xarray as xr
import matplotlib.pyplot as plt
from typing import Optional, Literal, Dict, Any

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f


@hdf5_tool(
    category="analysis",
    keywords=["visualize", "plot", "graph", "chart", "image", "heatmap", "line plot", "histogram", "contour", "display"],
    use_cases=[
        "Creating plots of temperature data",
        "Visualizing 2D sensor arrays",
        "Generating histograms of distributions",
        "Plotting time series data",
    ],
)
def visualize(
    filepath: str,
    object_path: str,
    plot_type: Optional[Literal['auto', 'line', 'scatter', 'hist', 'pcolormesh',
                                 'imshow', 'contour', 'contourf']] = 'auto',
    hdf5_slices: Optional[Dict[int, Any]] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    xscale: str = 'linear',
    yscale: str = 'linear',
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    save_path: Optional[str] = None
) -> dict:
    """
    Generate a visualization (PNG image) of an HDF5 dataset.

    Automatically selects appropriate plot types (line, histogram, heatmap, contour, etc.)
    based on data dimensionality and semantic metadata, or accepts explicit plot type
    specification. Supports memory-efficient HDF5-level slicing for large datasets,
    dimension selection, and axis scaling. Designed to work with get_object_metadata()
    and read_semantic_metadata() to make informed visualization decisions.

    VISUALIZATION DECISION GUIDE:

    1. ALWAYS call get_object_metadata() first to understand data shape and dimensions
    2. READ semantic metadata with read_semantic_metadata() to inform plot choices

    PLOT TYPE SELECTION (based on dimensionality):
    - 1D data: Use 'line' for time series or ordered data, 'hist' for distributions
    - 2D data: Use 'pcolormesh' for regular grids, 'imshow' for image-like data (faster),
               'contour'/'contourf' for smooth fields
    - Use 'auto' (default) to let the function choose based on dimensionality

    HDF5 SLICING (memory efficient, applied before loading):
    - Use hdf5_slices to select subsets: {0: slice(0, 100), 1: slice(None), 2: 50}
    - Keys are dimension indices (0=first dim, 1=second dim, etc.)
    - Values can be: slice objects, integers, or slice(None) for entire dimension
    - Example: {0: slice(0, 1000), 2: 0} selects first 1000 along dim 0, index 0 along dim 2

    DIMENSION SELECTION (xarray-level):
    - x/y: Specify which dimension names to plot on each axis
    - For 2D plots, if x/y not specified, uses first two dimensions by default
    - For 1D plots, uses the single dimension automatically

    SCALE SELECTION:
    - Use xscale='log' or yscale='log' when data spans multiple orders of magnitude
    - Check SMD for hints about appropriate scaling (e.g., logarithmic phenomena)

    WORKFLOW:
    1. get_object_metadata(filepath, object_path) - understand shape/dims
    2. read_semantic_metadata(filepath, object_path) - understand meaning
    3. Decide plot_type based on dimensionality and semantics
    4. Use hdf5_slices if dataset is very large (check shape in metadata)
    5. Call visualize() with appropriate parameters

    Args:
        filepath: Path to HDF5 file
        object_path: Path to dataset within file (e.g., "/group/dataset")
        plot_type: Type of plot ('auto' infers from dimensions)
        hdf5_slices: Dict mapping dimension index to slice/int for subsetting
        x: Name of dimension to plot on x-axis (optional, auto-selected if None)
        y: Name of dimension to plot on y-axis (optional, auto-selected if None)
        xscale: Scale for x-axis ('linear', 'log', 'symlog', 'logit')
        yscale: Scale for y-axis ('linear', 'log', 'symlog', 'logit')
        xlim: Tuple of (min, max) for x-axis limits
        ylim: Tuple of (min, max) for y-axis limits
        save_path: Path to save plot image (if None, auto-generates name)

    Returns:
        Dictionary with:
        - success: bool
        - save_path: str (path where plot was saved)
        - plot_info: dict with plot details
        - error: str (only if success=False)
    """
    try:
        # Load data from HDF5
        with h5py.File(filepath, 'r') as f:
            if object_path not in f:
                return {
                    "success": False,
                    "error": f"Object '{object_path}' not found in file"
                }

            obj = f[object_path]

            if not isinstance(obj, h5py.Dataset):
                return {
                    "success": False,
                    "error": f"Object '{object_path}' is not a dataset (type: {type(obj).__name__})"
                }

            # Apply HDF5 slicing if provided
            if hdf5_slices:
                # Build slice tuple for h5py
                ndims = len(obj.shape)
                slice_tuple = [slice(None)] * ndims

                for dim_idx, slice_spec in hdf5_slices.items():
                    if dim_idx >= ndims:
                        return {
                            "success": False,
                            "error": f"Slice dimension index {dim_idx} out of range for {ndims}D dataset"
                        }
                    slice_tuple[dim_idx] = slice_spec

                data = obj[tuple(slice_tuple)]
                sliced_shape = data.shape
            else:
                data = obj[:]
                sliced_shape = data.shape

            # Get dimension names from attributes if available
            dim_names = []
            if 'dimensions' in obj.attrs:
                dim_names_attr = obj.attrs['dimensions']
                if isinstance(dim_names_attr, bytes):
                    dim_names = dim_names_attr.decode('utf-8').split(',')
                elif isinstance(dim_names_attr, str):
                    dim_names = dim_names_attr.split(',')
                elif hasattr(dim_names_attr, '__iter__'):
                    dim_names = [str(d) for d in dim_names_attr]

            # Generate default dimension names if not available
            if not dim_names or len(dim_names) != len(sliced_shape):
                dim_names = [f"dim_{i}" for i in range(len(sliced_shape))]

        # Convert to xarray DataArray
        da = xr.DataArray(data, dims=dim_names)

        # Determine plot type based on dimensionality
        ndims = len(sliced_shape)

        if plot_type == 'auto':
            if ndims == 1:
                plot_type = 'line'
            elif ndims == 2:
                plot_type = 'pcolormesh'
            else:
                return {
                    "success": False,
                    "error": f"Cannot auto-select plot type for {ndims}D data. Must be 1D or 2D, or use hdf5_slices to reduce dimensions."
                }

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot based on type
        if plot_type == 'line':
            if ndims != 1:
                return {
                    "success": False,
                    "error": f"Line plot requires 1D data, got {ndims}D"
                }
            da.plot.line(ax=ax)

        elif plot_type == 'hist':
            da.plot.hist(ax=ax)

        elif plot_type == 'scatter':
            if ndims < 2:
                return {
                    "success": False,
                    "error": f"Scatter plot requires at least 2D data or specified x/y variables"
                }
            # For scatter, need to specify x and y
            if x and y:
                da.plot.scatter(x=x, y=y, ax=ax)
            else:
                return {
                    "success": False,
                    "error": "Scatter plot requires x and y parameters to be specified"
                }

        elif plot_type in ['pcolormesh', 'imshow', 'contour', 'contourf']:
            if ndims != 2:
                return {
                    "success": False,
                    "error": f"{plot_type} requires 2D data, got {ndims}D. Use hdf5_slices to reduce dimensions."
                }

            plot_kwargs = {}
            if x:
                plot_kwargs['x'] = x
            if y:
                plot_kwargs['y'] = y

            if plot_type == 'pcolormesh':
                da.plot.pcolormesh(ax=ax, **plot_kwargs)
            elif plot_type == 'imshow':
                da.plot.imshow(ax=ax, **plot_kwargs)
            elif plot_type == 'contour':
                da.plot.contour(ax=ax, **plot_kwargs)
            elif plot_type == 'contourf':
                da.plot.contourf(ax=ax, **plot_kwargs)

        else:
            return {
                "success": False,
                "error": f"Unknown plot type: {plot_type}"
            }

        # Apply axis scaling
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        # Apply axis limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        # Generate save path if not provided
        if not save_path:
            import os
            base_name = object_path.replace('/', '_').strip('_')
            save_path = f"{base_name}_plot.png"

        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return {
            "success": True,
            "save_path": save_path,
            "plot_info": {
                "plot_type": plot_type,
                "dimensions": dim_names,
                "shape": sliced_shape,
                "xscale": xscale,
                "yscale": yscale
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error creating visualization: {str(e)}"
        }
