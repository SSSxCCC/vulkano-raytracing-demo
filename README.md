# vulkano-raytracing-demo

An example of KHR raytracing using [vulkano](https://github.com/vulkano-rs/vulkano).

## Run

### Windows

```
cargo run -p minimal -F desktop
```

### Android

```
rustup target add aarch64-linux-android
cargo install cargo-ndk
cargo ndk -t arm64-v8a -o android-project/app/src/main/jniLibs/ build -p minimal
cd android-project
./gradlew build
./gradlew installDebug
```

## Demos

### minimal

![image](minimal.png)

### shapes

![image](shapes.png)

### weekend

![image](weekend.png)

### ray-query

![image](ray-query.png)
