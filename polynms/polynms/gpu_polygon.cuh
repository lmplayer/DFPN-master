#include<iostream>
#include <cmath>


__device__ float inline x_intersect(float x1, float y1, float x2, float y2,
                float x3, float y3, float x4, float y4)
{
    float num = (x1*y2 - y1*x2) * (x3-x4) -
            (x1-x2) * (x3*y4 - y3*x4);
    float den = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4);
    return num/(den+1e-9);
}

// Returns y-value of point of intersectipn of
// two lines
__device__ float inline y_intersect(float x1, float y1, float x2, float y2,
                float x3, float y3, float x4, float y4)
{
    float num = (x1*y2 - y1*x2) * (y3-y4) -
            (y1-y2) * (x3*y4 - y3*x4);
    float den = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4);
    return num/(den+1e-9);
}

// This functions clips all the edges w.r.t one clip
// edge of clipping area
__device__ void clip(float* poly_points, int& poly_size, float* buffer,
        float x1, float y1, float x2, float y2)
{
    int new_poly_size=0;
    // (ix,iy),(kx,ky) are the co-ordinate values of
    // the points
    for (int i = 0; i < poly_size; i++)
    {
        // i and k form a line in polygon
        int k = (i+1) % poly_size;
        float ix = poly_points[2*i], iy = poly_points[2*i+1];
        float kx = poly_points[2*k], ky = poly_points[2*k+1];

        // Calculating position of first point
        // w.r.t. clipper line
        float i_pos = (x2-x1) * (iy-y1) - (y2-y1) * (ix-x1);

        // Calculating position of second point
        // w.r.t. clipper line
        float k_pos = (x2-x1) * (ky-y1) - (y2-y1) * (kx-x1);

        // Case 1 : When both points are inside
        if (i_pos > 0 && k_pos > 0)
        {
            //Only second point is added
            buffer[2*new_poly_size] = kx;
            buffer[2*new_poly_size+1] = ky;
            new_poly_size++;
        }

        // Case 2: When only first point is outside
        else if (i_pos <= 0 && k_pos > 0)
        {
            // Point of intersection with edge
            // and the second point is added
            buffer[2*new_poly_size] = x_intersect(x1,
                            y1, x2, y2, ix, iy, kx, ky);
            buffer[2*new_poly_size+1] = y_intersect(x1,
                            y1, x2, y2, ix, iy, kx, ky);
            new_poly_size++;

            buffer[2*new_poly_size] = kx;
            buffer[2*new_poly_size+1] = ky;
            new_poly_size++;
        }

        // Case 3: When only second point is outside
        else if (i_pos > 0 && k_pos <= 0)
        {
            //Only point of intersection with edge is added
            buffer[2*new_poly_size] = x_intersect(x1,
                            y1, x2, y2, ix, iy, kx, ky);
            buffer[2*new_poly_size+1] = y_intersect(x1,
                            y1, x2, y2, ix, iy, kx, ky);
            new_poly_size++;
        }

        // Case 4: When both points are outside
        else
        {
            //No points are added
        }
    }

    // Copying new points into original array
    // and changing the no. of vertices
    poly_size = new_poly_size;
    for (int i = 0; i < 2*poly_size; i++)
    {
        poly_points[i] = buffer[i];
    }
}

// Implements Sutherland¨CHodgman algorithm
__device__ void suthHodgClip(float const* const poly_points, int poly_size,
                  float const * const clipper_points, int clipper_size,
                  float* new_points, int& new_size, float* buffer){
    new_size=poly_size;
    for(int i=0; i<2*poly_size; ++i){
        new_points[i]=poly_points[i];
    }
    
    //i and k are two consecutive indexes
    for (int i=0; i<clipper_size; i++){
        int k = (i+1) % clipper_size;

        // We pass the current array of vertices, it's size
        // and the end points of the selected clipper line
        clip(new_points, new_size, buffer, clipper_points[2*i],
            clipper_points[2*i+1], clipper_points[2*k],
            clipper_points[2*k+1]);
    }
}

// (X[i], Y[i]) are coordinates of i'th point.
__device__ float polygonArea(float const * const xy, int n){
    if (n<2)
        return 0.f;
    // Initialze area
    float area = 0.f;

    // Calculate value of shoelace formula
    int j = n - 1;
    for (int i = 0; i < n; i++){
        area += (xy[2*j] + xy[2*i]) * (xy[2*j+1] - xy[2*i+1]);
        j = i;  // j is previous vertex to i
    }

    // Return absolute value
    return std::fabs(area / 2.f);
}
