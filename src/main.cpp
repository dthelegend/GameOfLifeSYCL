#include <sycl/sycl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#define LIVING_COLOR 0xFF
#define DEAD_COLOR 0x00

int main(int argc, char const *argv[])
{
    cv::Mat cv_image;

    if (argc > 2) {
        std::cout << "Too many arguments!" << std::endl;
        
        return 128;
    }
    if (argc == 2) {
        cv_image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

        if (cv_image.data == NULL) {
            std::cout << "Failed to load image" << std::endl;

            return 128;
        }
    }
    else {
        cv_image = cv::Mat(480, 480, CV_8UC1, cv::Scalar(DEAD_COLOR));
        
        cv_image.at<uchar>(0,1)=LIVING_COLOR;
        cv_image.at<uchar>(1,2)=LIVING_COLOR;
        cv_image.at<uchar>(2,0)=LIVING_COLOR;
        cv_image.at<uchar>(2,1)=LIVING_COLOR;
        cv_image.at<uchar>(2,2)=LIVING_COLOR;
    }

    while (true) {
        cv::imshow("Conway's Game of Life", cv_image);

        int k = cv::waitKey(0);

        // Create an image buffer
        {
            sycl::queue queue(sycl::default_selector_v);
            sycl::range<2> image_range{(size_t) cv_image.rows, (size_t) cv_image.cols};
            
            size_t mat_size = cv_image.total() * cv_image.elemSize();

            uchar * buffer_copy = (uchar *) malloc(mat_size);
            memcpy(buffer_copy, cv_image.data, mat_size);

            sycl::buffer sycl_read_buffer(buffer_copy, image_range);
            sycl::buffer sycl_write_buffer(cv_image.data, image_range);

            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor ReadAccessor{sycl_read_buffer, cgh, sycl::read_only};
                sycl::accessor WriteAccessor{sycl_write_buffer, cgh, sycl::write_only};

                cgh.parallel_for(image_range, [=](sycl::id<2> WorkItemId) {
                    int count = 0;

                    for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j) {
                        if (i == 0 && j == 0)
                            continue;

                        sycl::id<2> c_id(WorkItemId[0] + i, WorkItemId[1] + j);

                        // Make sure to handle boundary conditions
                        if (c_id[0] >= 0 && c_id[0] < image_range[0] && c_id[1] >= 0 && c_id[1] < image_range[1]) {
                            count += ReadAccessor[c_id] == LIVING_COLOR ? 1 : 0;
                        }
                    }

                    if (count == 3) {
                        WriteAccessor[WorkItemId] = LIVING_COLOR;
                    }
                    else if (count == 2) {
                        WriteAccessor[WorkItemId] = ReadAccessor[WorkItemId];
                    }
                    else {
                        WriteAccessor[WorkItemId] = DEAD_COLOR;
                    }
                });
            });
        }
    }

    return 0;
}
