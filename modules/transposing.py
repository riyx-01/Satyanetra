from PIL import Image

def main():
    try:
        # Relative path to your image
        img = Image.open("C:\\Users\\riyat\\Downloads\\UploadFromMobile\\IMG_0667.jpeg")
        
        # Transpose the image (flip horizontally)
        transposed_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Save the transposed image
        transposed_img.save("transposed_redr.jpg")
        print("Image transposed and saved successfully.")
    except IOError as e:
        print("An error occurred while processing the image:", e)

if __name__ == "__main__":
    main()
