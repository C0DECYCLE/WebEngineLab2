console.log(PIXI);

async function init() {
    const app = new PIXI.Application();
    await app.init({
        background: "#FFFFFF",
        resizeTo: window,
    });
    console.log(app);
    document.body.appendChild(app.canvas);

    const texture = await PIXI.Assets.load("./src/iosbug/quest.png");

    for (let i = 0; i < 10000; i++) {
        const bunny = new PIXI.Sprite(texture);
        bunny.x = Math.random() * 4000;
        bunny.y = Math.random() * 4000;
        app.stage.addChild(bunny);
    }
}
