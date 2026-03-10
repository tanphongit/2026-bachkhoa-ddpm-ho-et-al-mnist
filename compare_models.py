from PIL import Image, ImageDraw, ImageFont
import os


def add_title(image, title, font=None):
    """Add title on top of one image."""
    w, h = image.size
    title_height = 40

    canvas = Image.new("RGB", (w, h + title_height), "white")
    canvas.paste(image, (0, title_height))

    draw = ImageDraw.Draw(canvas)

    if font is None:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    x = (w - text_w) // 2
    y = (title_height - text_h) // 2
    draw.text((x, y), title, fill="black", font=font)

    return canvas


def main():
    # ===== Input images =====
    diffusion_path = "outputs/diffusion/final_samples.png"
    vae_path = "outputs/vae/samples/samples_epoch_05.png"
    gan_path = "outputs/gan/samples/samples_epoch_05.png"

    # ===== Output =====
    out_dir = "report/figures"
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "comparison_diffusion_vae_gan.png")

    # ===== Load images =====
    diffusion_img = Image.open(diffusion_path).convert("RGB")
    vae_img = Image.open(vae_path).convert("RGB")
    gan_img = Image.open(gan_path).convert("RGB")

    # ===== Resize all to same height =====
    target_height = min(diffusion_img.height, vae_img.height, gan_img.height)

    def resize_keep_ratio(img, target_h):
        w, h = img.size
        new_w = int(w * target_h / h)
        return img.resize((new_w, target_h))

    diffusion_img = resize_keep_ratio(diffusion_img, target_height)
    vae_img = resize_keep_ratio(vae_img, target_height)
    gan_img = resize_keep_ratio(gan_img, target_height)

    # ===== Add titles =====
    diffusion_img = add_title(diffusion_img, "Diffusion")
    vae_img = add_title(vae_img, "VAE")
    gan_img = add_title(gan_img, "GAN")

    # ===== Create final canvas =====
    margin = 20
    total_width = diffusion_img.width + vae_img.width + gan_img.width + margin * 4
    max_height = max(diffusion_img.height, vae_img.height, gan_img.height) + margin * 2

    canvas = Image.new("RGB", (total_width, max_height), "white")

    x = margin
    for img in [diffusion_img, vae_img, gan_img]:
        y = (max_height - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + margin

    # ===== Save =====
    canvas.save(output_path)
    print(f"Saved comparison image to: {output_path}")


if __name__ == "__main__":
    main()