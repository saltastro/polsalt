#!/usr/bin/env python


import os, sys, pyfits
# import wlcal
# import traceline

# import pysalt.mp_logging
# import logging
import numpy
import scipy, scipy.interpolate

import matplotlib.pyplot as pl



def compute_spline_sky_spectrum(all_skies, 
                                n_basepoints=100,
                                N_min=10,
                                show_plot_range=None):
    """
    Take all sky datapoint tuples (wavelength, flux) and fit a spline to them.

    Parameters
    ----------

    all_skies : 2-d numpy array

        first col is wavelength, 2nd col is flux

    n_basepoints : int

        how many base points for the spline fit. Recommended are 
    
    N_min : int

        set the minimum number of datapoints required for a basepoint to be fit.
        Setting this to >= 5 avoids problems where the spline fitting attempts
        to fit a data point to no or not enough data, resulting in truncated and/or
        incomplete spline fits and subsequently to problems when using the spline
        for sky subtraction

    Returns
    -------

    spline_fit : generator function for the spline

        spline_fit can be used to interpolate/compute the spline by calling
        sky = spline_fit(wavelength);
        Returns None if no spline fit could be computed


    """

#    logger = logging.getLogger("FitSplineSky")
#    logger.info("Fitting sky spectrum with spline")

    #
    # Find minimum and maximum wavelength range so we can compute 
    # a spline fit to the sky spectrum
    #
    wl_min = numpy.min(all_skies[:,0])
    wl_max = numpy.max(all_skies[:,0])
#    logger.info("Sky wavelength range: %f -- %f" % (wl_min, wl_max))

    # compute basepoints
    # skip first and last to ensure we do not exceed the input range
    basepoints = numpy.linspace(wl_min, wl_max, n_basepoints+2)[1:-1]

#    logger.info("Using basepoints in range %f -- %f for spline fit" % (
#            basepoints[0], basepoints[-1]))


    # -- For debugging --
    # numpy.save("spline_x", all_skies[:,0])
    # numpy.save("spline_y", all_skies[:,1])
    # numpy.save("spline_xy", all_skies)
    # numpy.savetxt("spline_xy.txt", all_skies, "%.3f %.2f")
    # numpy.savetxt("spline_t", basepoints)

    # Now reject all basepoints with insufficient datapoints close to them
    # require at least N datapoints
    kdtree = scipy.spatial.cKDTree(all_skies[:,0].reshape((-1,1)))
    search_radius = basepoints[1] - basepoints[0]
    nearest_neighbor, i = kdtree.query(x=basepoints.reshape((-1,1)), 
                                       k=N_min, # only find 1 nearest neighbor
                                       p=1, # use linear distance
                                       distance_upper_bound=search_radius)
    neighbor_count = numpy.sum( numpy.isfinite(nearest_neighbor), axis=1)
#    print neighbor_count.shape
    
#    numpy.savetxt("neighbor_count", 
#                  numpy.append(basepoints.reshape((-1,1)),
#                               neighbor_count.reshape((-1,1)), axis=1)
#                  )

    #
    # Now eliminate all basepoints with not enough data points for proper fitting
    #
    basepoints = basepoints[neighbor_count >= N_min]
    
    #
    # Now attempt the actual spline fit
    #
    sky_spectrum_spline = None
    try:
        sky_spectrum_spline = scipy.interpolate.LSQUnivariateSpline(
            x=all_skies[:,0], 
            y=all_skies[:,1], 
            t=basepoints, 
            w=None, # no weights (for now)
            bbox=[wl_min, wl_max], 
            k=3, # use a cubic spline fit
            )

        #
        # For debugging, compute the spline fit at all basepoints and dump to txt file
        #
        ss = numpy.append(basepoints.reshape((-1,1)),
                          sky_spectrum_spline(basepoints).reshape((-1,1)),
                          axis=1)
#        numpy.savetxt("skyspectrum.knots", sky_spectrum_spline.get_knots())
#        numpy.savetxt("skyspectrum.coeffs", sky_spectrum_spline.get_coeffs())
#        numpy.savetxt("skyspectrum.txt", ss)
    except:
#        logger.critical("Error with spline-fitting the sky-spectrum")
#        pysalt.mp_logging.log_exception()

        pass

    if (not show_plot_range == None and not sky_spectrum_spline == None):
        data2plot = (all_skies[:,0] >= show_plot_range[0]) & (all_skies[:,0] <= show_plot_range[1])
        plot_x = all_skies[data2plot][:,0]
        plot_y = all_skies[data2plot][:,1]

        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(show_plot_range) #(5850,5950))
        #ax.set_ylim((0,3800))
    
        ax.scatter(plot_x, plot_y, linewidths=0) #,s=1,marker=",")
        ax.scatter(basepoints,numpy.ones_like(basepoints)*400, linewidths=0, c='r')
        ax.plot(basepoints, sky_spectrum_spline(basepoints), 'g-', linewidth=2)

        fig.show()
        pl.show()


    return sky_spectrum_spline


def make_2d_skyspectrum(obj_data, 
                        wls_2d,
                        sky_regions=None, 
                        oversample_factor=2.0,
                        slitprofile=None):
    """
    Compute a full 2-D sky spectrum, including curvature, based on the input 
    HDUList and the 2-D wavelength solution created from an appropriate ARC 
    spectrum using the same setup.

    Parameters
    ----------

    hdulist : pyfits.HDUList

        multi-extension FITS HDUList of input object frame. 

    wls_2d : numpy 2d array

        two-dimensional numpy array with wavelengths for each pixel

    sky_regions : numpy (N,2) array

        list of y-positions (from,to) marking which positions along the slit 
        (vertical bands if image is displayed in ds9) to be used for extracting 
        the sky spectrum.

    oversample_factor : float

        ratio between number of spline basepoints to be used for interpolating 
        the sky spectrum and the number of pixels in spectral direction in the
        input object frame.


    Returns
    -------

    2-d sky spectrum as numpy ndarray.


    """

#    logger = logging.getLogger("Make2DSkySpec")

    #
    # Now extract some sky-spectrum from the specified y-range 
    # Make copy to make sure we don't accidently change the data
    #
#    obj_data = hdulist['SCI'].data #numpy.array(hdulist['SCI'].data)
    if (type(slitprofile) == numpy.ndarray and slitprofile.ndim == 1):
        # If we have a valid slitprofile (i.e. a 1-d numpy array)
        obj_data /= slitprofile.reshape((-1,1))

        
        
        
    # Remember: Both FITS data __AND__ WLS_2D data are in [y,x] ordering
    all_skies = None

    obj_masked = numpy.empty(obj_data.shape)
    obj_masked[:,:] = numpy.NaN

    for idx, sky_region in enumerate(sky_regions):
#        logger.debug("Adding sky-region: y = %4d ... %4d" % (sky_region[0], sky_region[1]))

        #print obj_data.shape,  sky_region[0], sky_region[1]
        data_region = obj_data[sky_region[0]:sky_region[1], :]
        wls_region = wls_2d[sky_region[0]:sky_region[1], :]
        
        obj_masked[sky_region[0]:sky_region[1], :] = obj_data[sky_region[0]:sky_region[1], :]

        # Now merge data and wavelengths
        # this gives us a 2-D array, shape N,2 with WL in the zero-th column, 
        # and fluxes in the first
        #print data_region.shape, wls_region.shape
        data_wls = numpy.append(wls_region.reshape((-1,1)),
                                data_region.reshape((-1,1)),
                                axis=1)
        # For now dump this data to file
        # if (idx == 0):
        #     numpy.savetxt("wl+data__%d-%d.dump" % (sky_region[0],sky_region[1]),
        #                   data_wls)

        all_skies = data_wls if all_skies == None else \
            numpy.append(all_skies, data_wls, axis=0)

    #
    # XXXXXXXX
    # Change this to add masked region as separate extension
    #
#    pyfits.HDUList([pyfits.PrimaryHDU(header=hdulist['SCI'].header,
#                                      data=obj_masked)]).writeto("obj_masked.fits", clobber=True)

    #
    # Exclude all points with NaNs in either wavelength or flux
    #
    good_pixel = numpy.isfinite(all_skies[:,0]) & numpy.isfinite(all_skies[:,1])
    all_skies = all_skies[good_pixel]
    
    #
    # also sort all pixels to be ascending in wavelength, otherwise the spline 
    # fitting will crap out with some "Interior knots t must satisfy "
    # Schoenberg-Whitney conditions" error message that does not seem to make 
    # any sense
    #
    wl_sort = numpy.argsort(all_skies[:,0])
    all_skies = all_skies[wl_sort]

#    numpy.savetxt("allskies", all_skies[::10])

    ############################################################################
    #
    # Now we have a full list of wavelengths and presumed sky fluxes
    #
    ############################################################################


    #
    # Fit a spline to the spectrum. Use N times as many basepoints as there 
    # are pixels in spectral direction in the original FITS data
    #
    #N_oversample = 1.1 #2.
    N_original = obj_data.shape[1]
#    logger.info("Oversampling %d input pixels by a factor of %.1f" % (
#            N_original, oversample_factor))
    n_basepoints = N_original * oversample_factor
    sky_spectrum_spline = compute_spline_sky_spectrum(
        all_skies, 
        n_basepoints=n_basepoints,
        N_min=10,
        show_plot_range=None, #[5800,6000],
        )


    #
    # Now with the spline fit to the sky-spectrum, we can compute the 2-D sky 
    # spectrum for the full input frame, including the curvature in the spectral 
    # dimension which we haven't compensated for yet.
    #
#    logger.info("Computing full-frame, 2-D sky spectrum, incl. curvature ...")
    sky_2d = sky_spectrum_spline(wls_2d.ravel())
    sky_2d = sky_2d.reshape(wls_2d.shape)
    
    # For now, write the sky spectrum to FITS so we can have a look at it in ds9
#    pyfits.HDUList([pyfits.PrimaryHDU(data=sky_2d)]).writeto("sky_2d.fits", clobber=True)

    return sky_2d



