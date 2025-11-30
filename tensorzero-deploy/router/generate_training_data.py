#!/usr/bin/env python3
"""
Generate synthetic training data for the neural router.
Creates realistic video generation scenarios with performance metrics.
"""

import json
import random
import numpy as np
from collections import defaultdict

# Configuration
N_SAMPLES = 5000
OUTPUT_DIR = "."

# Realistic video production project descriptions (tailored for script_generator function)
DESCRIPTIONS = [
    # DIY Tutorials
    "Create a step-by-step video on building a custom PC from scratch",
    "Film a beginner's guide to home brewing craft beer",
    "Produce a tutorial on restoring vintage furniture",
    "Make a video showing how to install solar panels on a roof",
    "Demonstrate professional cake decorating techniques",
    "Show the process of making artisanal cheese at home",
    "Create a guide to urban foraging for edible plants",
    "Film a comprehensive car maintenance checklist",
    "Produce a video on composting for beginners",
    "Demonstrate making natural skincare products",
    # Educational Content
    "Explain the science behind climate change in simple terms",
    "Create a video series on world history through artifacts",
    "Teach basic coding concepts with visual examples",
    "Produce an educational video on sustainable farming",
    "Explain quantum physics using everyday analogies",
    "Create a documentary-style video on ocean conservation",
    "Teach sign language basics for communication",
    "Produce a video on the evolution of technology",
    "Explain economic principles through real-world examples",
    "Create an educational series on mental health awareness",
    # Creative Projects
    "Film a time-lapse of creating a digital artwork from concept to finish",
    "Produce a short film about local community heroes",
    "Create a video showcasing handmade jewelry design",
    "Film the process of writing and illustrating a children's book",
    "Produce a music video for an original song",
    "Create a stop-motion animation of a day in the life of objects",
    "Film a fashion design process from sketch to garment",
    "Produce a video art piece using found footage",
    "Create a documentary on urban street art",
    "Film a creative cooking challenge with unusual ingredients",
    # Technical and Advanced
    "Demonstrate advanced woodworking joinery techniques",
    "Create a video on programming a robot for automation",
    "Produce a guide to 3D printing complex mechanical parts",
    "Film the assembly of a custom gaming setup",
    "Create a tutorial on drone cinematography",
    "Produce a video on ethical hacking basics",
    "Demonstrate CNC machining for custom parts",
    "Create a guide to building smart home devices",
    "Film the process of developing a mobile app",
    "Produce a video on renewable energy systems",
    # Lifestyle and Personal
    "Create a video on minimalist living principles",
    "Produce a guide to meditation and mindfulness practices",
    "Film a journey through different cultural cuisines",
    "Create a video on personal finance management",
    "Produce a series on home organization and decluttering",
    "Film the process of learning a new language",
    "Create a video on sustainable fashion choices",
    "Produce a guide to urban gardening in small spaces",
    "Film a personal development journey",
    "Create a video on work-life balance strategies",
    # Entertainment and Fun
    "Produce a comedy sketch about everyday absurdities",
    "Create a viral challenge video with a twist",
    "Film a magic trick tutorial with professional tips",
    "Produce a podcast-style video interview",
    "Create a dance tutorial for social media trends",
    "Film a cooking competition with celebrity judges",
    "Produce a travel vlog from unique destinations",
    "Create a music performance cover with visuals",
    "Film a gaming walkthrough with commentary",
    "Produce a comedy roast of common household items",
]

# Content mode distribution
CONTENT_MODES = {
    "kid": 0.4,  # 40% kid-friendly content
    "adult": 0.6,  # 60% general/adult content
}

# Duration options with realistic weights
DURATIONS = [60, 90, 120, 150, 180, 210, 240, 300]  # seconds
DURATION_WEIGHTS = [0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.03, 0.02]

# LLM variant performance profiles (tailored to TensorZero config models with real benchmarks)
VARIANT_PROFILES = {
    "cerebras_fast": {  # cerebras_llama70b: Instant Speed - Best for Drafting
        "quality_range": (0.7, 0.9),  # Strong quality for drafting
        "latency_range": (0.2, 0.6),  # Very fast (Cerebras optimized)
        "cost_multiplier": 1.2,  # Moderate cost
    },
    "sambanova": {  # sambanova_405b: Massive Reasoning - Best for Judging
        "quality_range": (0.8, 0.98),  # Excellent quality for complex reasoning
        "latency_range": (0.8, 2.0),  # Medium speed (large model)
        "cost_multiplier": 2.0,  # Higher cost
    },
    "groq": {  # groq_llama33: Reliable Speed
        "quality_range": (0.7, 0.9),  # Good quality
        "latency_range": (0.1, 0.5),  # Very fast (Groq TTFT <0.5s)
        "cost_multiplier": 1.7,  # Medium-high cost
    },
    "mistral": {  # mistral_large: High Quality European Model
        "quality_range": (0.75, 0.95),  # High quality
        "latency_range": (0.5, 1.5),  # Medium speed
        "cost_multiplier": 1.0,  # Balanced cost
    },
    "deepseek": {  # deepseek_v3: Best Value/Logic
        "quality_range": (0.75, 0.95),  # Strong quality for logic
        "latency_range": (1.0, 2.5),  # Slower but cost-effective
        "cost_multiplier": 0.3,  # Very low cost
    },
}


def generate_base_samples(n_samples):
    """Generate base training samples with realistic distributions."""
    samples = []

    for _ in range(n_samples):
        description = random.choice(DESCRIPTIONS)
        content_mode = random.choices(
            list(CONTENT_MODES.keys()), weights=list(CONTENT_MODES.values())
        )[0]
        duration = random.choices(DURATIONS, weights=DURATION_WEIGHTS)[0]

        samples.append(
            {"description": description, "content_mode": content_mode, "duration": duration}
        )

    return samples


def simulate_variant_performance(sample, variant_name):
    """Simulate how a specific LLM variant would perform on this sample with enhanced realism."""
    profile = VARIANT_PROFILES[variant_name]

    # Base quality from profile
    base_quality = random.uniform(*profile["quality_range"])

    # Kid content gets quality boost from some models
    if sample["content_mode"] == "kid" and variant_name in ["groq", "cerebras_fast"]:
        base_quality += 0.08

    # Complex descriptions need higher quality models
    complexity = len(sample["description"].split()) / 50  # Normalize
    if complexity > 1.0 and variant_name in ["sambanova", "deepseek"]:
        base_quality += 0.06

    # Duration impact: longer videos may require more processing
    duration_factor = sample["duration"] / 180  # Normalize around 3 min
    if duration_factor > 1.0:
        base_quality -= 0.02  # Slight penalty for very long content
        latency_penalty = duration_factor * 0.1
    else:
        latency_penalty = 0

    # Add Gaussian noise for realism
    noise = np.random.normal(0, 0.03)
    quality = min(1.0, max(0.0, base_quality + noise))

    # Latency simulation with duration impact
    base_latency = random.uniform(*profile["latency_range"])
    latency = base_latency + latency_penalty
    latency = max(0.1, latency)  # Minimum latency

    # Cost calculation (scaled for realism)
    cost = latency * profile["cost_multiplier"] * 0.02

    return {
        "quality_score": quality,
        "latency_ms": latency * 1000,  # Convert to ms
        "cost": cost,
        "variant": variant_name,
    }


def generate_comparative_training_data(samples):
    """Generate training data by comparing all variants on each sample."""
    training_data = []

    for sample in samples:
        variant_results = []

        # Test all variants on this sample
        for variant_name in VARIANT_PROFILES.keys():
            result = simulate_variant_performance(sample, variant_name)
            variant_results.append(result)

        # Calculate utility scores and find best variant
        # Utility = Quality² / log(Latency) - Cost * weight - favors quality over speed and cost
        cost_weight = (
            0.3  # Higher weight for cost sensitivity (matches config emphasis on free/paid)
        )
        for result in variant_results:
            utility = (result["quality_score"] ** 2) / np.log1p(result["latency_ms"]) - (
                result["cost"] * cost_weight
            )
            result["utility"] = utility

        best_variant = max(variant_results, key=lambda x: x["utility"])

        # Create training sample in ClickHouse format
        training_sample = {
            "input": {
                "project_description": sample["description"],
                "content_mode": sample["content_mode"],
                "target_duration": sample["duration"],
            },
            "variant_name": best_variant["variant"],
            "quality_score": best_variant["quality_score"],
            "latency_ms": best_variant["latency_ms"],
            "utility_score": best_variant["utility"],
        }

        training_data.append(training_sample)

    return training_data


def create_variant_map():
    """Create variant name to index mapping."""
    return {str(i): name for i, name in enumerate(VARIANT_PROFILES.keys())}


def calculate_statistics(training_data):
    """Calculate statistics for data validation."""
    variant_counts = defaultdict(int)
    quality_scores = []

    for sample in training_data:
        variant_counts[sample["variant_name"]] += 1
        quality_scores.append(sample["quality_score"])

    return {
        "total_samples": len(training_data),
        "variant_distribution": dict(variant_counts),
        "quality_stats": {
            "mean": float(np.mean(quality_scores)),
            "std": float(np.std(quality_scores)),
            "min": float(np.min(quality_scores)),
            "max": float(np.max(quality_scores)),
        },
    }


def main():
    print("🎯 Generating synthetic training data for neural router...")

    # Generate base samples
    print(f"📝 Creating {N_SAMPLES} base samples...")
    base_samples = generate_base_samples(N_SAMPLES)

    # Generate comparative training data
    print("🤖 Simulating LLM variant performance...")
    training_data = generate_comparative_training_data(base_samples)

    # Create variant mapping
    variant_map = create_variant_map()

    # Calculate statistics
    stats = calculate_statistics(training_data)

    # Save files
    print("💾 Saving training data...")

    with open(f"{OUTPUT_DIR}/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    with open(f"{OUTPUT_DIR}/variant_map.json", "w") as f:
        json.dump(variant_map, f, indent=2)

    with open(f"{OUTPUT_DIR}/data_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("✅ Training data generation complete!")
    print(f"   📊 {stats['total_samples']} samples generated")
    print(f"   🎯 Best variants: {stats['variant_distribution']}")
    print(
        f"   📈 Quality: {stats['quality_stats']['mean']:.3f} ± {stats['quality_stats']['std']:.3f}"
    )


if __name__ == "__main__":
    main()
