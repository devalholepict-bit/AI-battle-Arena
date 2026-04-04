import express from 'express';
import runGraph from "./ai/grap.ai.js"

const app = express();


app.get('/', async  (req, res) => {

    const result = await runGraph("Write an code for Factorial function in js")

    res.json(result)
})



export default app;