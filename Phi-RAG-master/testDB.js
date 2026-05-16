import { supabase } from "./supabaseClient.js";

async function testDB() {
  const { data, error } = await supabase
    .from("chats")
    .select("*");

  console.log("DATA:", data);
  console.log("ERROR:", error);
}

testDB();