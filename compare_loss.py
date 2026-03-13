from PIL import Image, ImageDraw, ImageFont
import os


def add_title(img, title):
    w, h = img.size
    title_h = 40

    canvas = Image.new("RGB", (w, h + title_h), "white")
    canvas.paste(img, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (w - text_w) // 2
    y = (title_h - text_h) // 2

    draw.text((x, y), title, fill="black", font=font)

    return canvas


def main():

    # ===== Paths =====
    ddpm_path = "outputs/diffusion/loss_curve.png"
    vae_path = "outputs/vae/loss_curve.png"
    gan_g_path = "outputs/gan/loss_curve_G.png"
    gan_d_path = "outputs/gan/loss_curve_D.png"

    out_dir = "report/figures"
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, "loss_comparison.png")

    # ===== Load images =====
    ddpm = Image.open(ddpm_path).convert("RGB")
    vae = Image.open(vae_path).convert("RGB")
    gan_g = Image.open(gan_g_path).convert("RGB")
    gan_d = Image.open(gan_d_path).convert("RGB")

    # ===== Add titles =====
    ddpm = add_title(ddpm, "DDPM Loss")
    vae = add_title(vae, "VAE Loss")
    gan_g = add_title(gan_g, "GAN Generator Loss")
    gan_d = add_title(gan_d, "GAN Discriminator Loss")

    imgs = [ddpm, vae, gan_g, gan_d]

    # ===== Resize same size =====
    min_w = min(i.width for i in imgs)
    min_h = min(i.height for i in imgs)

    imgs = [i.resize((min_w, min_h)) for i in imgs]

    # ===== Create grid 2x2 =====
    margin = 20

    canvas_w = min_w * 2 + margin * 3
    canvas_h = min_h * 2 + margin * 3

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    positions = [
        (margin, margin),
        (min_w + margin * 2, margin),
        (margin, min_h + margin * 2),
        (min_w + margin * 2, min_h + margin * 2)
    ]

    for img, pos in zip(imgs, positions):
        canvas.paste(img, pos)

    # ===== Save =====
    canvas.save(output_path)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()