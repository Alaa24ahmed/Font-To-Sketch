import aspose.words as aw

doc = aw.Document()
builder = aw.DocumentBuilder(doc)

shape = builder.insert_image("output\\arabic\\حصان_Horse_0\\ArefRuqaa-Regular_حصان_1.svg")
# pageSetup = builder.page_setup
# shape.color_mode = aw.ColorMode.GRAYSCALE

# Access the image data
# image_data = shape.image_data

# # Set the background color to white
# image_data.background_shape_color = aw.drawing.Color.white

# # Save the document with the modified image data
# # doc.save("Output.docx")
import aspose.pydrawing as pydraw
# # Convert the document to PNG
# image_data.save("output\\arabic\\حصان_Horse_0\\ArefRuqaa-Regular_حصان_1.png")
fill = shape.fill
# fill.fore_theme_color = aw.themes.ThemeColor.DARK1
# fill.back_theme_color = aw.themes.ThemeColor.BACKGROUND2
shape.fill.back_color = pydraw.Color.white
pageSetup = builder.page_setup
pageSetup.page_width = shape.width
pageSetup.page_height = shape.height
pageSetup.top_margin = 0
pageSetup.left_margin = 0
pageSetup.bottom_margin = 0
pageSetup.right_margin = 0
# Note: do not use "BackThemeColor" and "BackTintAndShade"for font fill.
# if fill.back_tint_and_shade == 0:
#     fill.back_tint_and_shade = 0.2


# shape = builder.insert_image(f"{experiment_dir}/{font}_{word}_{letter}.svg")
# shape.fill_color = "light_blue"
doc.save("output\\arabic\\حصان_Horse_0\\ArefRuqaa-Regular_حصان_1.png")


# import pyvips

# image = pyvips.Image.new_from_file("output\\arabic\\حصان_Horse_0\\ArefRuqaa-Regular_حصان_1.svg", dpi=300)
# image.write_to_file("output\\arabic\\حصان_Horse_0\\ArefRuqaa-Regular_حصان_1.png")

import pydiffvg
import torch
def save_image(img, filename, gamma=1):
    # check_and_create_dir(filename)
    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, filename, gamma=gamma)

# canvas_width, canvas_height, shapes_word, shape_groups_word= pydiffvg.svg_to_scene("output\\arabic\\حصان_Horse_0\\ArefRuqaa-Regular_حصان_1.svg")
# render = pydiffvg.RenderFunction.apply
# scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes_word, shape_groups_word)
# img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
# img = img[:, :, 3:4] * img[:, :, :3] + \
#             torch.ones(img.shape[0], img.shape[1], 3, device="cuda:0") * (1 - img[:, :, 3:4])
# img = img[:, :, :3]
# save_image(img, "output\\arabic\\حصان_Horse_0\\ArefRuqaa-Regular_حصان_1.png")
