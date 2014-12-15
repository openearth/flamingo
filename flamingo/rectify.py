'''Ortho-rectify images based on ground control points (GCP)

Usage:
    rectify-image <image> <gcpfile> [--dist-model=NAME] [--dist-coefs=VALUES] [--verbose]

Positional arguments:
    image               image to be rectified
    gcpfile             file containing GCP's in image (UVXYZ)

Options:
    -h, --help           show this help message and exit
    --dist-model=NAME    name of distortion model to use [default: OPENCV]
    --dist-coefs=VALUES  coefficients used for distortion model [default: 0,0,0,0]
    --size=SIZE          size of output figure [default: 30,20]
    --rotation=ANGLE     rotate resulting image [default: 0]
    --translation=DIST   translate resulting image [default: 0,0]
    --maxdistance=DIST   maximum distance from origin included in plot [default: 1e4]
    --verbose            print logging messages
'''

import os
import rectification

def run_rectification(img, gcpfile, dist_model='OPENCV', dist_coefs=[0,0,0,0],
                      figsize=(30,20), rotation=0, translation=0, max_distance=1e4):

    # undistort gcp's
    UV = [undistort([u], [v], rectification_data=r1[s][c]) for u,v in r1[s][c]['UV']]

    # find homography
    H = rectification.find_homography(UV, r1[s][c]['XYZ'], r1[s][c]['K'])

    # undistort image
    u, v = rectification.get_pixel_coordinates(img)
    u, v = undistort(u, v, rectification_data=r1[s][c])

    # rectify image
    x, y = rectification.rectify_coordinates(u, v, H)
    
    # plot image
    fig, axs = rectification.plot.plot_rectified([-x], [y], [img], 
                                                 figsize=figsize,
                                                 max_distance=max_distance,
                                                 axs=axs,
                                                 rotation=rotation,
                                                 translation=translation)

    return fig, axs


def main():
    import docopt

    arguments = docopt.docopt(__doc__)

    if arguments['--verbose']:
        logging.basicConfig()
        logging.root.setLevel(logging.NOTSET)

    run_rectification(
        arguments['<image>'],
        arguments['<gcpfile>'],
        dist_model=arguments['--dist-model'],
        dist_coefs=[float(x) for x in arguments['--dist-coefs'].split(',')],
        figsize=[float(x) for x in arguments['--size'].split(',')],
        rotation=float(arguments['--rotation']),
        translation=[float(x) for x in arguments['--translation'].split(',')],
        max_distance=float(arguments['--maxdistance'])
    )


if __name__ == '__main__':
    main()
