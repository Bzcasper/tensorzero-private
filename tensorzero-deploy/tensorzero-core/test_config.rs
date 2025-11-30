use std::path::PathBuf;
use tensorzero_core::config::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing config loading with router...");
    
    let config_path = PathBuf::from("../tensorzero-deploy/config/tensorzero.toml");
    println!("Loading config from: {:?}", config_path.canonicalize()?);
    
    let config = Config::load_from_path(&config_path)?;
    println!("✅ Config loaded successfully!");
    
    if let Some(router) = &config.gateway.router {
        println!("Router config found:");
        println!("  Model path: {:?}", router.model_path);
        println!("  Tokenizer path: {:?}", router.tokenizer_path);
        
        // Check if files exist
        let model_path = config_path.parent().unwrap().join(&router.model_path);
        let tokenizer_path = config_path.parent().unwrap().join(&router.tokenizer_path);
        
        println!("  Model file exists: {}", model_path.exists());
        println!("  Tokenizer file exists: {}", tokenizer_path.exists());
    } else {
        println!("❌ No router config found!");
        return Err("Router config missing".into());
    }
    
    println!("✅ Router integration config test passed!");
    Ok(())
}
