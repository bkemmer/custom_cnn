import matplotlib,pyplot as plt
import cv2

def plot_rectangle(detected_faces, image, title='Image', cmap_type='gray', kwargs={'lw': 20.}):
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(image, cmap=cmap_type)
    plt.title(title)
    for (column, row, width, height) in detected_faces:
        rect = Rectangle(
                (column, row),
                width = width,
                height = height,
                fill=False,
                edgecolor='r',
                
                )
        # Add the patch to the Axes
        ax.add_patch(rect)
#     plt.axis('off')
    plt.show()


def plt_two_imgs(img_a, img_b, cmap='gray', normalize=False):
    f = plt.figure(figsize=(12, 8))
    f.add_subplot(1,2, 1)
    if normalize:
        plt.imshow(img_a, vmin=np.min(img_matrix), vmax=np.max(img_matrix), cmap=cmap)
    else:
        plt.imshow(img_a, cmap=cmap)
    f.add_subplot(1,2, 2)
    if normalize:
        plt.imshow(img_b, vmin=np.min(img_matrix), vmax=np.max(img_matrix), cmap=cmap)
    else:
        plt.imshow(img_b, cmap=cmap)
    
    plt.show(block=True)
    

def plt_img(img_matrix, title='Image', normalize=False):
    '''
    Parameters: 
        - img_matrix: (ndarray)
        - title: (string)
    Output:
        - image plot
    '''
    if normalize:
        plt.imshow(img_matrix, vmin=np.min(img_matrix), vmax=np.max(img_matrix), cmap='gray')
    else:
        io.imshow(img_matrix)
    plt.title(title)
    plt.show()