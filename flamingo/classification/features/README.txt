WORKFLOW FOR FEATURE EXTRACTION
===============================

all_stats = []

# list all feature blocks
print flamingo.classification.features.blocks.list_blocks()

# loop over all images
for fname in image_files:

    # read image
    img = plt.imread(fname)

    # extract all features from all blocks
    df, features_in_block = flamingo.classification.features.blocks.extract_blocks(img, segments)

    # make features scale invariant
    df = flamingo.classification.features.scaleinvariant.scale_features(img, df)

    # linearize features
    df = flamingo.classification.features.linearize(df)

    # compute feature statistics
    stats = flamingo.classification.features.normalize.compute_feature_stats(df)

    # collect feature statistics
    all_stats.append(stats)

# aggregate feature statistics
stats = flamingo.classification.features.normalize.aggregate_feature_stats(all_stats)

# normalize features for a single image
df = flamingo.classification.features.normalize.normalize_features(df, stats)

# be awesome
