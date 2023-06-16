from PIL import Image, ImageDraw, ImageFont


def calculate_whitespace(words, image_width, font, line):
    total_w = 0
    for word in words[line*10:(line+1)*10]:
        word_w = font.getlength(word)
        total_w += word_w
    whitespace = image_width - total_w
    word_amnt = len(words[line*10:(line+1)*10]) if len(words[line*10:(line+1)*10]) <= 10 else 10
    if word_amnt == 0:
        return 0
    whitespace_per_word = whitespace // (word_amnt + 1)
    return whitespace_per_word


def max_word_h(words, font):
    word_h = 0
    for word in words:
        ascent, descent = font.getmetrics()
        (width, baseline), (offset_x, offset_y) = font.font.getsize(word)
        word_h = max(ascent - offset_y, word_h)
    return word_h


def create_weighted_words_image(words, weights, emotion, title, image_width=1000,
                                background_color=(255, 255, 255),
                                text_color=(0, 0, 0), font_size=20,
                                font_path='Montserrat-Medium.otf', glow_file='glow.png'):
    # Create a blank image
    image_height = 70 + 70 * (len(words)/10 if len(words) % 10 == 0 else len(words)/10 + 1)
    image = Image.new('RGBA', (image_width, int(image_height)), background_color)
    draw = ImageDraw.Draw(image)
    glow = Image.open(glow_file)

    # Set up the font
    font = ImageFont.truetype(font_path, font_size)

    # Find the maximum weight
    max_weight = max(weights)
    min_weight = min(weights)

    # Set initial x and y for word placement
    title_w = font.getlength(title)
    x = (image_width - title_w) // 2
    word_h = max_word_h([emotion], font)
    y = (70 - word_h) // 2
    word_position = (x, y)
    draw.text(word_position, title, fill=(0, 0, 0, 1), font=font)
    y += 70
    line = 0
    whitespace = calculate_whitespace(words, image_width, font, line)
    x = whitespace
    min_opacity = 70
    max_opacity = 255

    # Draw the weighted words
    for i, (word, weight) in enumerate(zip(words, weights)):
        word_w = font.getlength(word)
        # Calculate the opacity of the word based on its weight
        opacity = int((max_opacity - min_opacity) * ((weight - min_weight) / (max_weight - min_weight)) + min_opacity)\
            if min_weight != max_weight else max_opacity

        word_position = (int(x), int(y))
        glow_position = (int(x-8), int(y-3))
        glow = glow.resize((int(word_w + 15), int(word_h + 20)))

        image.paste(glow, glow_position, mask=glow.split()[3].point(lambda z: int(z * opacity / 255.)))
        draw.text(word_position, word, fill=text_color, font=font)
        if i % 10 == 9:
            y += 70
            line += 1
            whitespace = calculate_whitespace(words, image_width, font, line)
            x = whitespace
        else:
            x += word_w + whitespace

    # Save the image
    print(title + "\n(saved)")
    return image
